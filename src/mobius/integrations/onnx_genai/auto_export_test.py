# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the onnx-genai write_onnx_genai_config dispatcher."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import onnx_ir as ir
import pytest
import yaml

from mobius._configs import QuantizationConfig
from mobius._model_package import ModelPackage
from mobius.integrations.onnx_genai import write_onnx_genai_config
from mobius.integrations.onnx_genai._test_support import (
    _Cfg,
    _decoder_package,
    _model,
    _value,
    _vlm_package,
)
from mobius.integrations.onnx_genai.auto_export import (
    _ddim_alpha_schedule,
    _flow_match_euler_schedule,
    _looks_like_image_edit,
    _looks_like_video_diffusion,
)
from mobius.integrations.onnx_genai.inference_metadata import SchedulerConfig
from mobius.integrations.onnx_genai.workflow_metadata import (
    HierarchicalAudioWorkflowConfig,
    build_decoder_workflow_metadata,
    build_hierarchical_audio_workflow_metadata,
)


@dataclasses.dataclass
class _Int4Cfg(_Cfg):
    dtype: ir.DataType = ir.DataType.FLOAT16
    quantization: QuantizationConfig = dataclasses.field(
        default_factory=lambda: QuantizationConfig(bits=4, quant_method="rtn")
    )


class _DiffusionPkg(dict):
    pass


def _diffusion_package(*, text: bool = False):
    latent = ["batch", 4, "height", "width"]
    denoiser_inputs = [
        _value("sample", ir.DataType.FLOAT, latent),
        _value("timestep", ir.DataType.FLOAT, ["batch"]),
    ]
    components = {}
    if text:
        denoiser_inputs.append(
            _value(
                "encoder_hidden_states",
                ir.DataType.FLOAT,
                ["batch", "prompt_sequence", 32],
            )
        )
        components["text_encoder"] = _model(
            "text_encoder",
            [_value("input_ids", ir.DataType.INT64, ["batch", "prompt_sequence"])],
            [
                (
                    "encoder_hidden_states",
                    ir.DataType.FLOAT,
                    ["batch", "prompt_sequence", 32],
                )
            ],
        )
    denoiser = _model(
        "denoiser",
        denoiser_inputs,
        [("noise_pred", ir.DataType.FLOAT, latent)],
    )
    vae = _model(
        "vae_decoder",
        [_value("latent", ir.DataType.FLOAT, latent)],
        [("image", ir.DataType.FLOAT, ["batch", 3, "image_height", "image_width"])],
    )
    components.update({"denoiser": denoiser, "vae_decoder": vae})
    return ModelPackage(components)


def test_vibevoice_assets_and_revision_are_forwarded(monkeypatch, tmp_path):
    from mobius.integrations.onnx_genai import auto_export
    from mobius.models.vibevoice import VIBEVOICE_MODEL_ID, VIBEVOICE_REVISION
    from mobius.models.vibevoice_test import _make_tiny_models

    _, _, package = _make_tiny_models()
    calls: list[tuple[str, str | None]] = []

    def text_assets(output_dir, source, *, revision=None):
        calls.append(("text", revision))
        return {"tokenizer": str(Path(output_dir) / "tokenizer.json")}

    def runtime_assets(output_dir, source, names, *, revision=None):
        calls.append(("runtime", revision))
        assert names == ("processor_config.json", "generation_config.json")
        return {"processor_config": str(Path(output_dir) / "processor_config.json")}

    def audio_processor(output_dir, source, *, revision=None):
        calls.append(("audio", revision))
        return str(Path(output_dir) / "audio_processor.json")

    monkeypatch.setattr(auto_export, "_write_text_runtime_assets", text_assets)
    monkeypatch.setattr(auto_export, "_copy_runtime_assets", runtime_assets)
    monkeypatch.setattr(auto_export, "_write_hf_audio_processor", audio_processor)
    monkeypatch.setattr(
        auto_export,
        "_write_advisory_component_contract",
        lambda *args, **kwargs: {
            "inference_metadata": str(tmp_path / "inference_metadata.yaml")
        },
    )

    artifacts = write_onnx_genai_config(
        package,
        str(tmp_path),
        source=VIBEVOICE_MODEL_ID,
        revision=VIBEVOICE_REVISION,
    )

    assert calls == [
        ("text", VIBEVOICE_REVISION),
        ("runtime", VIBEVOICE_REVISION),
        ("audio", VIBEVOICE_REVISION),
    ]
    assert {
        "tokenizer",
        "processor_config",
        "audio_processor",
        "inference_metadata",
    } <= set(artifacts)


def test_vibevoice_asr_writes_processor_and_advisory_contract(monkeypatch, tmp_path):
    from mobius.models.vibevoice_asr_test import _make_models

    _, _, package = _make_models()
    monkeypatch.setattr(
        "mobius.integrations.onnx_genai.auto_export._write_text_runtime_assets",
        lambda *args, **kwargs: {},
    )

    artifacts = write_onnx_genai_config(package, str(tmp_path))

    processor = json.loads((tmp_path / "preprocessor_config.json").read_text())
    compatibility = json.loads((tmp_path / "runtime_compatibility.json").read_text())
    assert artifacts["processor_contract"] == str(tmp_path / "preprocessor_config.json")
    assert processor["target_sample_rate"] == 24_000
    assert (
        processor["speech_tok_compress_ratio"] == package.config.acoustic_tokenizer.hop_length
    )
    assert processor["encoder_final_chunk_input"] == "is_final_chunk"
    assert processor["acoustic_sampling"]["noise_scale_input"] == "acoustic_noise_scale"
    assert sorted(compatibility["components"]) == [
        "acoustic_encoder",
        "connectors",
        "decoder",
        "embedding",
        "semantic_encoder",
    ]
    assert compatibility["runtime_validation_status"] == "unsupported-by-tested-runtime"


def _video_diffusion_package() -> ModelPackage:
    latent = ["batch", "frames", 4, "height", "width"]
    transformer = _model(
        "transformer",
        [
            _value("sample", ir.DataType.FLOAT, latent),
            _value("timestep", ir.DataType.FLOAT, ["batch"]),
            _value(
                "encoder_hidden_states",
                ir.DataType.FLOAT,
                ["batch", "prompt_sequence", 32],
            ),
        ],
        [("noise_pred", ir.DataType.FLOAT, latent)],
    )
    text_encoder = _model(
        "text_encoder",
        [_value("input_ids", ir.DataType.INT64, ["batch", "prompt_sequence"])],
        [
            (
                "encoder_hidden_states",
                ir.DataType.FLOAT,
                ["batch", "prompt_sequence", 32],
            )
        ],
    )
    vae = _model(
        "vae_decoder",
        [
            _value(
                "latent_sample",
                ir.DataType.FLOAT,
                ["batch", 4, "latent_frames", "height", "width"],
            ),
            _value(
                "conv_cache.conv_in",
                ir.DataType.FLOAT,
                ["batch", 4, "cache_frames", "height", "width"],
            ),
        ],
        [
            ("sample", ir.DataType.FLOAT, ["batch", 3, "video_frames", "height", "width"]),
            (
                "conv_cache_out.conv_in",
                ir.DataType.FLOAT,
                ["batch", 4, "cache_frames", "height", "width"],
            ),
        ],
    )
    vae.metadata_props["mobius.conv_cache.spatial_scale.conv_cache.conv_in"] = "1"
    return ModelPackage(
        {
            "transformer": transformer,
            "text_encoder": text_encoder,
            "vae_decoder": vae,
        }
    )


class _MultimodalPkg(dict):
    config = _Cfg()


def test_dispatch_decoder(tmp_path):
    package = _decoder_package(_Int4Cfg())
    arts = write_onnx_genai_config(package, str(tmp_path), config=_Int4Cfg())
    with open(arts["inference_metadata"]) as handle:
        meta = yaml.safe_load(handle)
    workflow = meta["pipeline"]["workflow"]
    assert "ir_version" not in workflow["manifest"]
    assert "onnx_opsets" not in workflow["manifest"]
    assert workflow["components"]["token_sampler"]["contract"]["id"] == (
        "onnx-genai.token-sampler"
    )
    assert workflow["components"]["termination"]["contract"]["id"] == (
        "onnx-genai.termination-predicate"
    )
    assert workflow["steps"][0]["kind"] == "loop"
    application_inputs = {
        name
        for name, value in workflow["inputs"].items()
        if value["source"]["kind"] == "application"
    }
    assert application_inputs == {
        "request.prompt_lengths",
        "request.row_max_iterations",
        "request.rng_counter",
    }
    assert workflow["inputs"]["request.eos_ids"]["role"]["role"] == "eos_token_ids"
    assert workflow["inputs"]["request.eos_lengths"]["role"]["role"] == "eos_token_lengths"
    assert workflow["inputs"]["request.prompt_lengths"]["default"] == -1
    assert [node["component"] for node in workflow["steps"][0]["setup"]] == [
        "decoder_state_initializer",
        "model",
        "termination_batch_initializer",
        "last_token_logits",
    ]
    body = workflow["steps"][0]["steps"]
    assert [node["kind"] for node in body].count("emit") == 1
    assert next(node for node in body if node["kind"] == "emit")["value"] == "token.body"
    assert workflow["outputs"]["tokens"]["contract"]["shape"] == [
        "batch",
        "generated_sequence",
    ]
    assert workflow["steps"][0]["iteration"] == {
        "value": "loop.iteration",
        "contract": {"dtype": "int64", "rank": 1, "shape": [1]},
    }
    assert "iteration" not in workflow["state"]
    serialized = yaml.safe_dump(workflow)
    assert "initial_effects" not in serialized
    assert "read_effect" not in serialized
    assert "write_effect" not in serialized
    assert ".read" not in serialized
    assert "iteration_increment" not in workflow["components"]
    assert workflow["state"]["token"]["initializer"] == "initializer.token_slot"
    assert workflow["state"]["logits"] == {
        "contract": {
            "dtype": "float32",
            "rank": 2,
            "shape": ["batch", 128],
            "batch_layout": {"kind": "request_aligned", "axis": 0},
        },
        "scope": "invocation",
        "initializer": "decoder.setup.last_logits",
        "recurrence": {"kind": "invariant"},
    }
    assert (tmp_path / "policies" / "token_sampler.onnx").is_file()


def test_seeded_decoder_sampler_uses_request_controls_and_direct_kv_carry():
    workflow = build_decoder_workflow_metadata(
        _decoder_package(), _Cfg(), sampler="seeded_categorical"
    )["pipeline"]["workflow"]
    sampler = workflow["components"]["token_sampler"]
    assert sampler["application_overridable"] is True
    assert sampler["contract"]["version"] == "2"
    assert sampler["contract"]["parameters"] == {
        "mode": "seeded_stochastic",
        "batching": "per_row",
        "inactive_rows": "preserve",
    }
    assert sampler["contract"]["bindings"] == {
        "logits": "logits",
        "token": "token",
        "temperature": "temperature",
        "top_k": "top_k",
        "top_p": "top_p",
        "min_p": "min_p",
        "seed": "seed",
        "counter": "counter",
        "next_counter": "next_counter",
        "active": "active",
        "done": "done",
    }
    assert sampler["ports"]["inputs"]["logits"] == {
        "dtype": "float32",
        "rank": 2,
        "shape": ["batch", "vocabulary"],
        "batch_layout": {"kind": "request_aligned", "axis": 0},
    }
    assert sampler["ports"]["outputs"]["token"] == {
        "dtype": "int64",
        "rank": 1,
        "shape": ["batch"],
        "batch_layout": {"kind": "request_aligned", "axis": 0},
    }
    sampler_step = next(
        step
        for step in workflow["steps"][0]["steps"]
        if step.get("component") == "token_sampler"
    )
    assert sampler_step["inputs"] == {
        "logits": "logits",
        "temperature": "request.temperature",
        "top_k": "request.top_k",
        "top_p": "request.top_p",
        "min_p": "request.min_p",
        "seed": "request.seed",
        "counter": "rng_counter",
        "active": "active",
        "done": "done",
    }
    for name in ("temperature", "top_k", "top_p", "min_p"):
        assert workflow["inputs"][f"request.{name}"]["contract"]["shape"] == ["batch"]
    assert workflow["inputs"]["request.prompt_lengths"]["contract"]["shape"] == ["batch"]
    assert workflow["inputs"]["request.max_iterations"]["contract"]["shape"] == [1]
    assert workflow["inputs"]["request.eos_ids"]["contract"]["shape"] == [
        "batch",
        "num_eos",
    ]
    assert workflow["state"]["rng_counter"]["class"] == "semantic"
    assert workflow["state"]["rng_counter"]["initializer"] == "request.rng_counter"
    emit = next(node for node in workflow["steps"][0]["steps"] if node["kind"] == "emit")
    assert "row_ids" not in emit
    assert "emit_row_identity" not in workflow["manifest"]["capabilities"]
    assert set(
        workflow["serving"]["state_service"]["groups"]["decoder_cache"]["ports"]["model"]
    ) == {"cache_0", "cache_1"}
    assert workflow["components"]["termination"]["contract"]["version"] == "2"
    assert workflow["components"]["termination"]["contract"]["parameters"] == {
        "batching": "per_row",
        "inactive_rows": "preserve",
    }
    assert set(workflow["components"]["termination"]["contract"]["bindings"]) == {
        "tokens",
        "active",
        "eos_ids",
        "eos_lengths",
        "iteration",
        "max_iterations",
        "done",
        "next_active",
        "continue",
    }
    assert workflow["components"]["termination"]["ports"]["inputs"]["iteration"] == {
        "dtype": "int64",
        "rank": 1,
        "shape": [1],
    }
    assert workflow["components"]["token_state_update"]["contract"] == {
        "id": "onnx-genai.state-update",
        "version": "2",
        "bindings": {
            "current": "current",
            "update": "update",
            "active": "active",
            "done": "done",
            "next": "next",
        },
        "parameters": {
            "batching": "per_row",
            "inactive_rows": "preserve",
        },
    }
    assert not any("kv_update" in name for name in workflow["components"])


def test_dispatch_language_diffusion(tmp_path):
    package = ModelPackage(
        {
            "model": _model(
                "masked_denoiser",
                [_value("input_ids", ir.DataType.INT64, ["batch", "sequence"])],
                [
                    ("logits", ir.DataType.FLOAT, ["batch", "sequence", 128]),
                    ("proposed_tokens", ir.DataType.INT64, ["batch", "sequence"]),
                ],
            )
        },
        config=_Cfg(model_type="llada"),
    )
    artifacts = write_onnx_genai_config(
        package,
        str(tmp_path),
        num_inference_steps=12,
    )
    with open(artifacts["inference_metadata"]) as handle:
        metadata = yaml.safe_load(handle)
    pipeline = metadata["pipeline"]
    assert set(pipeline) == {"workflow"}
    assert pipeline["workflow"]["inputs"]["request.max_iterations"]["default"] == 12
    assert (tmp_path / "policies" / "masked_update.onnx").is_file()


def test_dispatch_diffusion(tmp_path):
    pkg = _diffusion_package()
    arts = write_onnx_genai_config(
        pkg,
        str(tmp_path),
        num_inference_steps=20,
        vae_filename="vae.onnx",
    )
    with open(arts["inference_metadata"]) as handle:
        meta = yaml.safe_load(handle)
    workflow = meta["pipeline"]["workflow"]
    assert workflow["steps"][0]["iteration"]["value"] == "loop.iteration"
    assert workflow["steps"][1]["component"] == "vae_decoder"
    assert "strategy" not in meta["pipeline"]
    assert (tmp_path / "policies" / "solver_step.onnx").is_file()
    assert (tmp_path / "policies" / "schedule_lookup.onnx").is_file()


def test_dispatch_video_diffusion_uses_typed_ddim(tmp_path, monkeypatch):
    package = _video_diffusion_package()
    assert _looks_like_video_diffusion(package)
    scheduler = SchedulerConfig(
        kind="ddim",
        prediction_type="v_prediction",
        timestep_spacing="trailing",
        rescale_betas_zero_snr=True,
        snr_shift_scale=3.0,
    )
    monkeypatch.setattr(
        "mobius.integrations.onnx_genai.auto_export._write_hf_tokenizer",
        lambda *args, **kwargs: None,
    )
    artifacts = write_onnx_genai_config(
        package,
        str(tmp_path),
        scheduler=scheduler,
        num_inference_steps=2,
        guidance_scale=6.0,
    )
    with open(artifacts["inference_metadata"]) as handle:
        metadata = yaml.safe_load(handle)
    workflow = metadata["pipeline"]["workflow"]
    assert workflow["outputs"]["video"]["contract"]["rank"] == 5
    loop = next(step for step in workflow["steps"] if step["kind"] == "loop")
    assert [node["component"] for node in loop["steps"]].count("transformer") == 2
    _, schedule = _ddim_alpha_schedule(scheduler, 2)
    assert schedule[0] < schedule[1] <= schedule[2]


def test_video_diffusion_requires_explicit_guidance(tmp_path):
    with pytest.raises(ValueError, match="video diffusion workflow must declare"):
        write_onnx_genai_config(
            _video_diffusion_package(),
            str(tmp_path),
            scheduler=SchedulerConfig(kind="ddim"),
            num_inference_steps=2,
        )


def test_video_diffusion_forwards_revision_to_semantic_config_loaders(tmp_path, monkeypatch):
    calls = []

    def fake_scheduler(source, *, revision=None):
        calls.append(("scheduler", source, revision))
        return SchedulerConfig(kind="ddim")

    def fake_vae_scaling(source, *, revision=None):
        calls.append(("vae", source, revision))
        return 0.13025

    monkeypatch.setattr(
        "mobius.integrations.onnx_genai.auto_export.load_diffusers_scheduler_config",
        fake_scheduler,
    )
    monkeypatch.setattr(
        "mobius.integrations.onnx_genai.auto_export.load_diffusers_vae_scaling_factor",
        fake_vae_scaling,
    )
    monkeypatch.setattr(
        "mobius.integrations.onnx_genai.auto_export._write_hf_tokenizer",
        lambda *args, **kwargs: None,
    )

    write_onnx_genai_config(
        _video_diffusion_package(),
        str(tmp_path),
        source="zai-org/CogVideoX-2b",
        revision="pinned-revision",
        num_inference_steps=2,
        guidance_scale=6.0,
    )

    assert calls == [
        ("scheduler", "zai-org/CogVideoX-2b", "pinned-revision"),
        ("vae", "zai-org/CogVideoX-2b", "pinned-revision"),
    ]


def test_single_diffusion_component_requires_explicit_vae(tmp_path):
    pkg = _DiffusionPkg({"transformer": object()})
    with pytest.raises(
        ValueError,
        match="diffusion workflow requires distinct denoiser and VAE decoder",
    ):
        write_onnx_genai_config(pkg, str(tmp_path), num_inference_steps=2)


def _image_edit_package(dtype: ir.DataType = ir.DataType.FLOAT):
    """Build a Qwen-Image-Edit-shaped package with the real port contract.

    Mirrors ``QwenImageTask``: rank-3 packed latents, a ``target_sequence_length``
    slice port, separate image/text rotary tables, and a VAE encoder/decoder pair.
    """
    tokens = ["batch", "image_sequence_length", 64]
    transformer = _model(
        "transformer",
        [
            _value("sample", dtype, tokens),
            _value("timestep", ir.DataType.FLOAT, ["batch"]),
            _value(
                "encoder_hidden_states",
                dtype,
                ["batch", "text_sequence_length", 32],
            ),
            _value(
                "encoder_hidden_states_mask",
                ir.DataType.BOOL,
                ["batch", "text_sequence_length"],
            ),
            _value("image_rotary_cos", dtype, ["image_sequence_length", 8]),
            _value("image_rotary_sin", dtype, ["image_sequence_length", 8]),
            _value("text_rotary_cos", dtype, ["text_sequence_length", 8]),
            _value("text_rotary_sin", dtype, ["text_sequence_length", 8]),
            _value("target_sequence_length", ir.DataType.INT64, [1]),
        ],
        [("noise_pred", dtype, tokens)],
    )
    vae_encoder = _model(
        "vae_encoder",
        [_value("pixel_values", dtype, ["batch", 3, 1, "height", "width"])],
        [("latent_sample", dtype, ["batch", 16, 1, "lheight", "lwidth"])],
    )
    vae_decoder = _model(
        "vae_decoder",
        [_value("latent_sample", dtype, ["batch", 16, 1, "lheight", "lwidth"])],
        [("image", dtype, ["batch", 3, 1, "height", "width"])],
    )
    return ModelPackage(
        {
            "transformer": transformer,
            "vae_encoder": vae_encoder,
            "vae_decoder": vae_decoder,
        }
    )


_FLOW_MATCH_SCHEDULER = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "base_image_seq_len": 256,
    "max_image_seq_len": 8192,
    "base_shift": 0.5,
    "max_shift": 0.9,
    "shift_terminal": 0.02,
    "time_shift_type": "exponential",
    "use_dynamic_shifting": True,
}


def _write_scheduler(source, config=None):
    (source / "scheduler").mkdir(parents=True, exist_ok=True)
    (source / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(_FLOW_MATCH_SCHEDULER if config is None else config), encoding="utf-8"
    )


def test_detects_image_edit_package_structurally():
    """Structural detection must not depend on model_type strings."""
    assert _looks_like_image_edit(_image_edit_package())
    # A plain latent-diffusion package has no VAE encoder and rank-4 samples.
    assert not _looks_like_image_edit(_diffusion_package(text=True))
    # A VAE pair alone is not enough without the target-slice denoiser port.
    pkg = _image_edit_package()
    del pkg["transformer"]
    assert not _looks_like_image_edit(pkg)


def test_flow_match_schedule_matches_diffusers():
    """Pin the schedule against the captured Qwen-Image-Edit-2509 reference.

    These are the timesteps ``QwenImageEditPlusPipeline`` produced for the
    1216x864 reference edit (``image_seq_len=4104``, 8 steps), divided by
    ``num_train_timesteps`` because the denoiser consumes normalized sigmas.
    """
    scheduler = SchedulerConfig(
        kind="flow_match_euler",
        use_dynamic_shifting=True,
        base_image_seq_len=256,
        max_image_seq_len=8192,
        base_shift=0.5,
        max_shift=0.9,
        shift_terminal=0.02,
        time_shift_type="exponential",
    )
    timesteps, sigmas = _flow_match_euler_schedule(scheduler, 8, 4104)
    expected = [
        1.0,
        0.9160475,
        0.8200923,
        0.7093592,
        0.5801504,
        0.4274216,
        0.2441077,
        0.02,
    ]
    assert timesteps == pytest.approx(expected, abs=1e-6)
    assert sigmas == pytest.approx([*expected, 0.0], abs=1e-6)


def test_flow_match_schedule_rejects_wrong_scheduler():
    with pytest.raises(ValueError, match="flow-match Euler scheduler"):
        _flow_match_euler_schedule(SchedulerConfig(kind="euler"), 4, 4104)


def test_dispatch_image_edit_emits_workflow(tmp_path):
    """A Qwen-Image-Edit-shaped package must dispatch to the image-edit workflow.

    Asserts the emitted pipeline actually performs the edit: encode the source
    image, run two guided denoiser passes per step, combine them with true CFG,
    and decode the target tokens back to pixels.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_scheduler(source)
    arts = write_onnx_genai_config(
        _image_edit_package(),
        str(output),
        source=str(source),
        num_inference_steps=8,
        image_seq_len=4104,
        guidance_scale=4.0,
        artifact_paths={"transformer": "models/transformer.onnx"},
    )
    with open(arts["inference_metadata"]) as handle:
        meta = yaml.safe_load(handle)
    workflow = meta["pipeline"]["workflow"]
    assert (
        workflow["components"]["transformer"]["implementation"]["artifact"]
        == "models/transformer.onnx"
    )

    loop = next(step for step in workflow["steps"] if step["kind"] == "loop")

    # The source image is encoded and packed once, in the loop's setup block.
    setup = [node["component"] for node in loop["setup"]]
    assert setup.index("vae_encoder") < setup.index("pack_latents")

    body = [node["component"] for node in loop["steps"]]
    # True CFG needs both a positive and a negative denoiser pass per step.
    assert body.count("transformer") == 2
    assert "true_cfg" in body
    assert "sequence_concat" in body
    assert body.index("true_cfg") < body.index("solver_step")
    assert loop["max_iterations"] == "request.max_iterations"

    # Source and target tokens are separate: only the target block is carried,
    # so the loop state's sequence axis is the target length, not the denoiser's
    # concatenated target+source length.
    assert [cell["cell"] for cell in loop["carried"]] == ["latent", "loop_0_active"]
    assert workflow["state"]["latent"]["contract"]["shape"] == [
        "batch",
        "target_sequence_length",
        64,
    ]

    tail = [step["component"] for step in workflow["steps"] if step["kind"] == "invoke"]
    assert tail == ["unpack_latents", "vae_decoder", "image_output_clamp"]

    # Positive and negative conditioning cannot share a sequence contract.
    inputs = workflow["inputs"]
    assert (
        inputs["request.positive_encoder_hidden_states"]["contract"]["shape"][1]
        == "positive_text_sequence_length"
    )
    assert (
        inputs["request.negative_encoder_hidden_states"]["contract"]["shape"][1]
        == "negative_text_sequence_length"
    )

    for policy in ("solver_step", "true_cfg", "pack_latents", "unpack_latents"):
        assert (output / "policies" / f"{policy}.onnx").is_file()


def test_image_edit_requires_explicit_guidance(tmp_path):
    source = tmp_path / "source"
    _write_scheduler(source)
    with pytest.raises(ValueError, match="image-edit workflow must declare"):
        write_onnx_genai_config(
            _image_edit_package(),
            str(tmp_path / "output"),
            source=str(source),
            num_inference_steps=8,
            image_seq_len=4104,
        )


def test_image_edit_forwards_revision_to_scheduler_loader(tmp_path, monkeypatch):
    calls = []

    def fake_scheduler(source, *, revision=None):
        calls.append((source, revision))
        return SchedulerConfig.from_diffusers(_FLOW_MATCH_SCHEDULER)

    monkeypatch.setattr(
        "mobius.integrations.onnx_genai.auto_export.load_diffusers_scheduler_config",
        fake_scheduler,
    )
    monkeypatch.setattr(
        "mobius.integrations.onnx_genai.auto_export._write_hf_tokenizer",
        lambda *args, **kwargs: None,
    )

    write_onnx_genai_config(
        _image_edit_package(),
        str(tmp_path),
        source="Qwen/Qwen-Image-Edit-2509",
        revision="pinned-revision",
        num_inference_steps=2,
        image_seq_len=4104,
        guidance_scale=4.0,
    )

    assert calls == [("Qwen/Qwen-Image-Edit-2509", "pinned-revision")]


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_image_edit_contract_uses_float_clamp_output(tmp_path, dtype):
    source = tmp_path / "source"
    output = tmp_path / "output"
    package = _image_edit_package(dtype)
    _write_scheduler(source)
    artifacts = write_onnx_genai_config(
        package,
        str(output),
        source=str(source),
        num_inference_steps=2,
        image_seq_len=4104,
        guidance_scale=4.0,
    )
    with open(artifacts["inference_metadata"]) as handle:
        workflow = yaml.safe_load(handle)["pipeline"]["workflow"]

    assert package["vae_decoder"].graph.outputs[0].dtype == dtype
    assert (
        package.policy_components["image_output_clamp"].model.graph.outputs[0].dtype
        == ir.DataType.FLOAT
    )
    assert workflow["outputs"]["image"]["contract"]["dtype"] == "float32"


def test_image_edit_requires_image_seq_len(tmp_path):
    """The schedule is resolution dependent, so it cannot be guessed."""
    source = tmp_path / "source"
    _write_scheduler(source)
    with pytest.raises(ValueError, match="requires image_seq_len"):
        write_onnx_genai_config(
            _image_edit_package(),
            str(tmp_path / "output"),
            source=str(source),
            num_inference_steps=8,
            guidance_scale=4.0,
        )


def test_image_edit_requires_scheduler_config(tmp_path):
    with pytest.raises(ValueError, match="requires the diffusers scheduler config"):
        write_onnx_genai_config(
            _image_edit_package(),
            str(tmp_path / "output"),
            num_inference_steps=8,
            image_seq_len=4104,
            guidance_scale=4.0,
        )


def test_dispatch_diffusion_emits_clip_tokenizer(tmp_path, monkeypatch):
    """Emit tokenizer.json for a text-conditioned diffusion package.

    The onnx-genai runners can then tokenize prompts from the package alone.
    """
    import os

    class _Backend:
        def save(self, path):
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("{}")

    class _Tokenizer:
        backend_tokenizer = _Backend()

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: _Tokenizer()
    )
    pkg = _diffusion_package(text=True)
    arts = write_onnx_genai_config(
        pkg,
        str(tmp_path),
        num_inference_steps=20,
        vae_filename="vae.onnx",
        text_encoder_filename="text_encoder.onnx",
        source="fake/model",
        guidance_scale=1.0,
    )
    assert "tokenizer" in arts
    assert os.path.basename(arts["tokenizer"]) == "tokenizer.json"
    assert os.path.isfile(arts["tokenizer"])


def test_dispatch_diffusion_tokenizer_skip_is_non_fatal(tmp_path, monkeypatch):
    """Skip the tokenizer artifact without failing when none can be loaded.

    If the source has no CLIP tokenizer (or transformers can't load one), the
    build still succeeds and simply omits the tokenizer artifact.
    """

    def _boom(*args, **kwargs):
        raise OSError("no tokenizer here")

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", _boom)
    pkg = _diffusion_package(text=True)
    arts = write_onnx_genai_config(
        pkg,
        str(tmp_path),
        num_inference_steps=20,
        text_encoder_filename="text_encoder.onnx",
        source="fake/model",
        guidance_scale=1.0,
    )
    assert "inference_metadata" in arts
    assert "tokenizer" not in arts


def test_dispatch_diffusion_auto_reads_scheduler_from_source(tmp_path):
    import json

    src = tmp_path / "ckpt"
    (src / "scheduler").mkdir(parents=True)
    (src / "scheduler" / "scheduler_config.json").write_text(
        json.dumps({"_class_name": "EulerDiscreteScheduler", "beta_schedule": "scaled_linear"})
    )
    out = tmp_path / "out"
    pkg = _diffusion_package()
    arts = write_onnx_genai_config(
        pkg,
        str(out),
        num_inference_steps=15,
        source=str(src),
    )
    with open(arts["inference_metadata"]) as handle:
        meta = yaml.safe_load(handle)
    components = meta["pipeline"]["workflow"]["components"]
    assert components["diffusion_schedule"]["ports"]["outputs"]["schedule"] == {
        "dtype": "float32",
        "rank": 1,
        "shape": [16],
    }
    schedule = ir.load(out / "policies" / "diffusion_schedule.onnx")
    assert list(schedule.graph.outputs[0].shape) == [16]


def test_diffusion_forwards_revision_to_semantic_config_loaders(tmp_path, monkeypatch):
    calls = []

    def fake_scheduler(source, *, revision=None):
        calls.append(("scheduler", source, revision))
        return SchedulerConfig(kind="euler")

    def fake_vae_scaling(source, *, revision=None):
        calls.append(("vae", source, revision))
        return 0.18215

    monkeypatch.setattr(
        "mobius.integrations.onnx_genai.auto_export.load_diffusers_scheduler_config",
        fake_scheduler,
    )
    monkeypatch.setattr(
        "mobius.integrations.onnx_genai.auto_export.load_diffusers_vae_scaling_factor",
        fake_vae_scaling,
    )

    write_onnx_genai_config(
        _diffusion_package(),
        str(tmp_path),
        source="nota-ai/bk-sdm-small",
        revision="pinned-revision",
        num_inference_steps=2,
    )

    assert calls == [
        ("scheduler", "nota-ai/bk-sdm-small", "pinned-revision"),
        ("vae", "nota-ai/bk-sdm-small", "pinned-revision"),
    ]


def test_dispatch_vision_multimodal_pipeline(tmp_path):
    pkg = _vlm_package()
    artifacts = write_onnx_genai_config(pkg, str(tmp_path))

    with open(artifacts["inference_metadata"]) as handle:
        metadata = yaml.safe_load(handle)

    pipeline = metadata["pipeline"]
    assert set(pipeline) == {"workflow"}
    workflow = pipeline["workflow"]
    assert workflow["manifest"]["adapter_abis"] == {"onnx-genai.image-preprocess": "1"}
    assert workflow["steps"][0]["setup"][0]["component"] == "image_preprocess"
    assert workflow["steps"][0]["setup"][1]["component"] == "vision_encoder"
    assert workflow["steps"][0]["setup"][4]["component"] == "embedding"
    assert workflow["steps"][0]["iteration"]["value"] == "loop.iteration"
    assert workflow["state"]["logits"]["contract"] == {
        "dtype": "float32",
        "rank": 2,
        "shape": ["batch", 128],
        "batch_layout": {"kind": "request_aligned", "axis": 0},
    }
    assert workflow["state"]["logits"]["initializer"] == "decoder.setup.last_logits"
    assert (tmp_path / "policies" / "token_sampler.onnx").is_file()


def test_workflow_vlm_rejects_kv_dtype_override(tmp_path):
    with pytest.raises(ValueError, match="kv_native_dtype overrides are unsupported"):
        write_onnx_genai_config(
            _vlm_package(),
            str(tmp_path),
            kv_native_dtype="bf16",
        )


def test_text_diffusion_requires_explicit_guidance_scale(tmp_path):
    with pytest.raises(ValueError, match="must declare its guidance"):
        write_onnx_genai_config(
            _diffusion_package(text=True),
            str(tmp_path),
        )


def test_guidance_scale_requires_a_text_conditioned_package(tmp_path):
    with pytest.raises(ValueError, match="requires a text-conditioned diffusion package"):
        write_onnx_genai_config(
            _diffusion_package(),
            str(tmp_path),
            guidance_scale=7.5,
        )


def test_dispatch_guided_diffusion_runs_the_denoiser_twice(tmp_path):
    arts = write_onnx_genai_config(
        _diffusion_package(text=True),
        str(tmp_path),
        num_inference_steps=4,
        guidance_scale=7.5,
    )
    with open(arts["inference_metadata"]) as handle:
        meta = yaml.safe_load(handle)
    workflow = meta["pipeline"]["workflow"]
    setup = workflow["steps"][0]["setup"]
    # The prompt and the negative prompt each get their own conditioning pass.
    assert [node.get("component") for node in setup].count("text_encoder") == 2
    negative = next(
        node
        for node in setup
        if node["inputs"].get("input_ids") == "request.negative_input_ids"
    )
    assert negative["outputs"]["encoder_hidden_states"] == "conditioning.unconditional"
    body = workflow["steps"][0]["steps"]
    denoiser_calls = [node for node in body if node.get("component") == "denoiser"]
    assert [node["outputs"]["noise_pred"] for node in denoiser_calls] == [
        "denoiser.unconditional",
        "denoiser.conditional",
    ]
    combine = next(node for node in body if node.get("component") == "guidance_combine")
    assert combine["inputs"] == {
        "unconditional": "denoiser.unconditional",
        "conditional": "denoiser.conditional",
        "scale": "request.guidance_scale",
    }
    assert combine["outputs"] == {"estimate": "denoiser.estimate"}
    assert (tmp_path / "policies" / "guidance_combine.onnx").is_file()


def test_dispatch_multistep_diffusion_carries_solver_history(tmp_path):
    import json

    source = tmp_path / "ckpt"
    (source / "scheduler").mkdir(parents=True)
    (source / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(
            {
                "_class_name": "DPMSolverMultistepScheduler",
                "beta_schedule": "scaled_linear",
                "algorithm_type": "dpmsolver++",
                "solver_order": 2,
                "solver_type": "midpoint",
                "lower_order_final": True,
                "final_sigmas_type": "zero",
            }
        )
    )
    arts = write_onnx_genai_config(
        _diffusion_package(text=True),
        str(tmp_path / "out"),
        num_inference_steps=4,
        guidance_scale=1.0,
        source=str(source),
    )
    with open(arts["inference_metadata"]) as handle:
        meta = yaml.safe_load(handle)
    workflow = meta["pipeline"]["workflow"]
    assert "history" in workflow["state"]
    carried = {carry["cell"]: carry["next"] for carry in workflow["steps"][0]["carried"]}
    assert "history.body" in carried.values()
    body = workflow["steps"][0]["steps"]
    solver = next(node for node in body if node.get("component") == "solver_step")
    history_cell = next(cell for cell, value in carried.items() if value == "history.body")
    assert solver["inputs"]["history"] == history_cell
    assert solver["outputs"]["next_history"] == "history.body"
    # A variance-preserving sampler feeds its state to the denoiser untouched, so
    # there is no model-input rescaling component at all.
    assert not any(node.get("component") == "model_input_scale" for node in body)
    latent_cell = next(cell for cell, value in carried.items() if value == "latent.body")
    denoiser = next(node for node in body if node.get("component") == "denoiser")
    assert denoiser["inputs"]["sample"] == latent_cell


def test_dispatch_audio_only_multimodal_pipeline(tmp_path):
    # The audio-only fusion shape used by speech-language ASR models such as
    # qwen3_asr and fun_asr: audio_encoder -> embedding fusion -> AR decoder.
    pkg = _vlm_package(audio=True)
    artifacts = write_onnx_genai_config(pkg, str(tmp_path))

    with open(artifacts["inference_metadata"]) as handle:
        metadata = yaml.safe_load(handle)

    setup = metadata["pipeline"]["workflow"]["steps"][0]["setup"]
    assert [node["component"] for node in setup[:3]] == [
        "image_preprocess",
        "vision_encoder",
        "audio_encoder",
    ]
    embedding = next(node for node in setup if node.get("component") == "embedding")
    assert embedding["inputs"]["audio_features"] == "audio.audio_features"


def test_audio_package_forwards_the_pinned_revision(tmp_path, monkeypatch):
    """A pinned revision must reach the asset writers, not just the build.

    Every runtime asset beside the graph — tokenizer, audio processor — is
    fetched from the Hub separately from the weights. If the revision stops
    being threaded, the package still builds and still validates, but its
    processor silently comes from whatever the branch tip happens to be, which
    is the failure a pin exists to prevent.
    """
    pkg = _vlm_package(audio=True)
    audio_processor = tmp_path / "audio_processor.json"
    audio_processor.write_text("{}")
    calls: list[tuple[str | None, str | None]] = []

    def fake_audio_processor(output_dir, source, *, revision=None):
        calls.append((source, revision))
        return str(audio_processor)

    monkeypatch.setattr(
        "mobius.integrations.onnx_genai.auto_export._write_hf_audio_processor",
        fake_audio_processor,
    )
    artifacts = write_onnx_genai_config(
        pkg,
        str(tmp_path),
        source="zai-org/GLM-ASR-Nano-2512",
        revision="pinned-revision",
    )
    assert artifacts["audio_processor"] == str(audio_processor)
    assert calls == [("zai-org/GLM-ASR-Nano-2512", "pinned-revision")]


def test_dispatch_vision_and_audio_multimodal_pipeline(tmp_path):
    pkg = _vlm_package(audio=True)
    artifacts = write_onnx_genai_config(pkg, str(tmp_path))

    with open(artifacts["inference_metadata"]) as handle:
        metadata = yaml.safe_load(handle)

    workflow = metadata["pipeline"]["workflow"]
    assert set(workflow["components"]) >= {
        "vision_encoder",
        "audio_encoder",
        "embedding",
        "decoder",
    }


class _FakeValue:
    def __init__(self, name: str, dtype: object | None = None) -> None:
        self.name = name
        self.dtype = dtype


class _FakeGraph:
    def __init__(self, input_names: list[str], output_names: list[str] | None = None) -> None:
        self.inputs = [_FakeValue(name) for name in input_names]
        self.outputs = [_FakeValue(name) for name in (output_names or [])]


class _FakeModel:
    """Minimal stand-in for an ir.Model exposing graph input/output names."""

    def __init__(self, input_names: list[str], output_names: list[str] | None = None) -> None:
        self.graph = _FakeGraph(input_names, output_names)


class _EncoderDecoderPkg(dict):
    config = _Cfg()


def _speech_package(*, encoder_mask: bool = False):
    """Whisper-shaped encoder/decoder package with a real cross-attention edge."""
    encoder_inputs = [
        _value("input_features", ir.DataType.FLOAT, ["batch", 80, "audio_seq_len"])
    ]
    encoder_outputs = [("encoder_hidden_states", ir.DataType.FLOAT, ["batch", 1500, 384])]
    decoder_inputs = [
        _value("decoder_input_ids", ir.DataType.INT64, ["batch", "sequence_len"]),
        _value("encoder_hidden_states", ir.DataType.FLOAT, ["batch", 1500, 384]),
        _value("position_ids", ir.DataType.INT64, ["batch", "sequence_len"]),
        _value(
            "past_key_values.0.key",
            ir.DataType.FLOAT,
            ["batch", 6, "past_sequence_len", 64],
        ),
        _value(
            "past_key_values.0.value",
            ir.DataType.FLOAT,
            ["batch", 6, "past_sequence_len", 64],
        ),
    ]
    decoder_outputs = [
        ("logits", ir.DataType.FLOAT, ["batch", "sequence_len", 51865]),
        ("present.0.key", ir.DataType.FLOAT, ["batch", 6, "total_sequence_len", 64]),
        ("present.0.value", ir.DataType.FLOAT, ["batch", 6, "total_sequence_len", 64]),
    ]
    if encoder_mask:
        encoder_inputs.append(
            _value("attention_mask", ir.DataType.INT64, ["batch", "audio_seq_len"])
        )
        encoder_outputs.append(("encoder_attention_mask", ir.DataType.INT64, ["batch", 1500]))
        decoder_inputs.insert(
            2, _value("encoder_attention_mask", ir.DataType.INT64, ["batch", 1500])
        )
    pkg = ModelPackage(
        {
            "encoder": _model("encoder", encoder_inputs, encoder_outputs),
            "decoder": _model("decoder", decoder_inputs, decoder_outputs),
        }
    )
    pkg.config = SimpleNamespace(eos_token_id=50257, max_position_embeddings=448)
    return pkg


def test_dispatch_speech_to_text_workflow(tmp_path):
    # Whisper-style ASR: the decoder consumes encoder_hidden_states (cross-attn).
    artifacts = write_onnx_genai_config(_speech_package(), str(tmp_path))

    with open(artifacts["inference_metadata"]) as handle:
        metadata = yaml.safe_load(handle)

    # The redesigned schema has no legacy pipeline description at all.
    assert "kv_cache" not in metadata
    assert not {"models", "dataflow", "strategy", "phases"}.intersection(metadata["pipeline"])
    workflow = metadata["pipeline"]["workflow"]
    assert {"encoder", "decoder"}.issubset(workflow["components"])

    # The encoder runs once in the loop prologue and its output persists as a
    # loop-invariant, request-aligned state cell the decoder reads every step.
    loop = next(step for step in workflow["steps"] if step["kind"] == "loop")
    setup_components = [node["component"] for node in loop["setup"]]
    # The encoder conditions the prefill, so it must precede the decoder.
    assert setup_components[0] == "encoder"
    assert setup_components.index("decoder") > 0
    cross = workflow["state"]["cross.encoder_hidden_states"]
    assert cross["contract"]["shape"] == ["batch", 1500, 384]
    assert cross["contract"]["batch_layout"] == {"kind": "request_aligned", "axis": 0}
    assert cross["initializer"] == "encoder.encoder_hidden_states"
    assert cross["recurrence"] == {"kind": "invariant"}
    carry = next(
        carry for carry in loop["carried"] if carry["cell"] == "cross.encoder_hidden_states"
    )
    assert carry["next"] == "cross.encoder_hidden_states"

    # The self-attention cache is a runtime-served state group; the invariant
    # cross state is not, because nothing appends to it.
    groups = workflow["serving"]["state_service"]["groups"]
    assert set(groups) == {"decoder_cache"}
    assert groups["decoder_cache"]["ports"]["decoder"]["cache_0"] == {
        "input": "past_key_values.0.key",
        "output": "present.0.key",
        "role": "key",
        "layer": 0,
    }


def test_dispatch_speech_to_text_rejects_kv_dtype_override(tmp_path):
    with pytest.raises(ValueError, match="derives KV state dtype"):
        write_onnx_genai_config(_speech_package(), str(tmp_path), kv_native_dtype="bf16")


def test_dispatch_speech_to_text_carries_encoder_mask_as_cross_state(tmp_path):
    artifacts = write_onnx_genai_config(_speech_package(encoder_mask=True), str(tmp_path))

    with open(artifacts["inference_metadata"]) as handle:
        metadata = yaml.safe_load(handle)

    workflow = metadata["pipeline"]["workflow"]
    # Every encoder output the decoder consumes becomes cross state, so an
    # encoder-side mask needs no special case.
    assert "cross.encoder_attention_mask" in workflow["state"]
    assert workflow["state"]["cross.encoder_attention_mask"]["contract"]["dtype"] == "int64"
    assert "encoder.input.attention_mask" in workflow["inputs"]


def test_dispatch_audio_codec_pipeline(tmp_path):
    encoder = _model(
        "encoder",
        [_value("waveform", ir.DataType.FLOAT, ["batch", 1, "audio_samples"])],
        [("codes", ir.DataType.INT64, ["batch", 16, "frames"])],
    )
    decoder = _model(
        "decoder",
        [_value("codes", ir.DataType.INT64, ["batch", 16, "frames"])],
        [("waveform", ir.DataType.FLOAT, ["batch", 1, "audio_samples"])],
    )
    pkg = ModelPackage({"encoder": encoder, "decoder": decoder})
    artifacts = write_onnx_genai_config(pkg, str(tmp_path))

    with open(artifacts["inference_metadata"]) as handle:
        metadata = yaml.safe_load(handle)

    assert "model" not in metadata
    assert not {"models", "dataflow", "strategy", "phases"}.intersection(metadata["pipeline"])
    workflow = metadata["pipeline"]["workflow"]
    assert workflow["steps"][0]["outputs"] == {"codes": "codec.codes"}
    assert workflow["steps"][1]["inputs"] == {"codes": "codec.codes"}
    assert workflow["outputs"]["waveform"]["stage"] == "post_adapter"


@pytest.mark.parametrize("package_kind", ["codec", "speech-to-text"])
def test_recognized_topology_superset_exports_advisory_contract(tmp_path, package_kind):
    if package_kind == "codec":
        encoder = _model(
            "encoder",
            [_value("waveform", ir.DataType.FLOAT, ["batch", 1, "audio_samples"])],
            [("codes", ir.DataType.INT64, ["batch", 16, "frames"])],
        )
        decoder = _model(
            "decoder",
            [_value("codes", ir.DataType.INT64, ["batch", 16, "frames"])],
            [("waveform", ir.DataType.FLOAT, ["batch", 1, "audio_samples"])],
        )
        pkg = _EncoderDecoderPkg({"encoder": encoder, "decoder": decoder})
    else:
        pkg = _EncoderDecoderPkg(dict(_speech_package()))
    pkg["auxiliary"] = _FakeModel(["aux_input"], ["aux_output"])

    artifacts = write_onnx_genai_config(pkg, str(tmp_path))
    compatibility = json.loads(
        Path(artifacts["runtime_compatibility"]).read_text(encoding="utf-8")
    )

    assert compatibility["runtime_validation_status"] == "unsupported-by-tested-runtime"
    assert set(compatibility["components"]) == {"encoder", "decoder", "auxiliary"}


@pytest.mark.parametrize("package_kind", ["codec", "speech-to-text"])
def test_recognized_topology_superset_with_malformed_component_fails(tmp_path, package_kind):
    if package_kind == "codec":
        pkg = _EncoderDecoderPkg(
            {
                "encoder": _FakeModel(["waveform"], ["codes"]),
                "decoder": _FakeModel(["codes"], ["waveform"]),
            }
        )
    else:
        pkg = _EncoderDecoderPkg(dict(_speech_package()))
    pkg["auxiliary"] = object()

    with pytest.raises(ValueError, match=r"auxiliary.*no graph contract"):
        write_onnx_genai_config(pkg, str(tmp_path))


def test_multi_decoder_tts_without_pre_embedder_raises_precise_not_implemented(tmp_path):
    # A nested multi-decoder TTS stack lacking the talker_step_embedder
    # pre-embedder cannot yet be mapped and must fail with a precise error.
    pkg = _EncoderDecoderPkg(
        {
            "talker": _FakeModel(["inputs_embeds"]),
            "code_predictor": _FakeModel(["inputs_embeds"]),
            "embedding": _FakeModel(["text_ids"]),
        }
    )
    with pytest.raises(NotImplementedError, match="nested generic workflow loops"):
        write_onnx_genai_config(pkg, str(tmp_path))


@pytest.mark.parametrize(
    (
        "hidden_size",
        "residual_codebooks",
        "codebook_size",
        "latent_channels",
        "condition_size",
        "channels",
        "source_rate",
        "target_rate",
        "semantic_start",
        "semantic_size",
        "stop_token",
        "flow_steps",
        "replace_from",
        "preserve_trailing",
    ),
    [
        (8, 7, 1024, 128, 2048, 2, 44100, 32000, 151675, 16384, 151670, 30, 1, 2),
        (12, 3, 257, 24, 96, 1, 48000, 24000, 700, 301, 699, 11, 0, 4),
    ],
)
def test_hierarchical_audio_package_is_not_misclassified_as_diffusion(
    tmp_path,
    hidden_size,
    residual_codebooks,
    codebook_size,
    latent_channels,
    condition_size,
    channels,
    source_rate,
    target_rate,
    semantic_start,
    semantic_size,
    stop_token,
    flow_steps,
    replace_from,
    preserve_trailing,
):
    float_type = ir.DataType.FLOAT
    pkg = ModelPackage(
        {
            "language_model": _model(
                "language_model",
                [
                    _value("inputs_embeds", float_type, [2, "sequence", hidden_size]),
                    _value("attention_mask", ir.DataType.INT64, [2, "context"]),
                    _value("position_ids", ir.DataType.INT64, [2, "sequence"]),
                    _value("past_key_values.0.key", float_type, [2, 1, "past_sequence", 4]),
                ],
                [
                    ("logits", float_type, [2, "sequence", 170000]),
                    ("last_hidden_state", float_type, [2, "sequence", hidden_size]),
                    ("present.0.key", float_type, [2, 1, "past_sequence", 4]),
                ],
            ),
            "language_model_embedding": _model(
                "language_model_embedding",
                [_value("input_ids", ir.DataType.INT64, [2, "sequence"])],
                [("inputs_embeds", float_type, [2, "sequence", hidden_size])],
            ),
            "language_model_semantic_embedding": _model(
                "language_model_semantic_embedding",
                [_value("semantic_codes", ir.DataType.INT64, [2, 1])],
                [("semantic_feedback_embedding", float_type, [2, 1, hidden_size])],
            ),
            "rvq_depth_decoder": _model(
                "rvq_depth_decoder",
                [_value("inputs_embeds", float_type, [2, "steps", hidden_size])],
                [("hidden_states", float_type, [2, "steps", hidden_size])],
            ),
            "rvq_depth_decoder_projection": _model(
                "rvq_depth_decoder_projection",
                [_value("hidden_states", float_type, [2, "steps", hidden_size])],
                [("projected_states", float_type, [2, "steps", hidden_size])],
            ),
            "rvq_depth_decoder_embedding": _model(
                "rvq_depth_decoder_embedding",
                [_value("code_ids", ir.DataType.INT64, [2, 1])],
                [("code_embeddings", float_type, [2, 1, hidden_size])],
            ),
            "rvq_depth_decoder_feedback_embedding": _model(
                "rvq_depth_decoder_feedback_embedding",
                [
                    _value(
                        "acoustic_codes",
                        ir.DataType.INT64,
                        [2, 1, residual_codebooks],
                    )
                ],
                [("acoustic_feedback_embedding", float_type, [2, 1, hidden_size])],
            ),
            "rvq_depth_decoder_heads": _model(
                "rvq_depth_decoder_heads",
                [_value("hidden_states", float_type, [2, "steps", hidden_size])],
                [
                    (
                        "all_codebook_logits",
                        float_type,
                        [residual_codebooks, 2, "steps", codebook_size],
                    )
                ],
            ),
            "condition_encoder": _model(
                "condition_encoder",
                [
                    _value(
                        "hidden_states",
                        float_type,
                        [1, "frames", hidden_size * (residual_codebooks + 1)],
                    )
                ],
                [
                    (
                        "encoder_hidden_states",
                        float_type,
                        [1, "latent_length", condition_size],
                    )
                ],
            ),
            "transformer": _model(
                "transformer",
                [
                    _value("hidden_states", float_type, [2, latent_channels, "latent_length"]),
                    _value("timestep", float_type, [2]),
                    _value(
                        "encoder_hidden_states",
                        float_type,
                        [2, "latent_length", condition_size],
                    ),
                ],
                [("sample", float_type, [2, latent_channels, "latent_length"])],
            ),
            "vocoder": _model(
                "vocoder",
                [_value("latents", float_type, [1, latent_channels, "latent_length"])],
                [("waveform", float_type, [1, channels, "samples"])],
            ),
        }
    )
    pkg.config = SimpleNamespace(
        component_configs={
            "condition_encoder": {
                "input_sampling_rate": 24000,
                "input_hop_length": 960,
                "output_hop_length": 512,
            },
            "vocoder": {"sampling_rate": source_rate},
        },
        workflow_config=HierarchicalAudioWorkflowConfig(
            components={
                "global_decoder": "language_model",
                "global_embedding": "language_model_embedding",
                "semantic_embedding": "language_model_semantic_embedding",
                "local_decoder": "rvq_depth_decoder",
                "local_projection": "rvq_depth_decoder_projection",
                "local_embedding": "rvq_depth_decoder_embedding",
                "local_feedback_embedding": "rvq_depth_decoder_feedback_embedding",
                "local_heads": "rvq_depth_decoder_heads",
                "condition_encoder": "condition_encoder",
                "flow_transformer": "transformer",
                "vocoder": "vocoder",
            },
            semantic_vocabulary_start=semantic_start,
            semantic_vocabulary_size=semantic_size,
            stop_token_id=stop_token,
            unconditional_token_id=stop_token - 1,
            semantic_guidance_scale=1.5,
            local_guidance_scale=1.5,
            flow_guidance_scale=1.7,
            sampling_top_k=50,
            chunk_frames=200,
            chunk_hop=100,
            flow_steps=flow_steps,
            carry_length=172,
            crop_left_latents=86,
            crop_right_latents=258,
            max_prompt_tokens=5000,
            max_audio_frames=9000,
            global_context=10240,
            target_sample_rate=target_rate,
            unconditional_replace_from=replace_from,
            unconditional_preserve_trailing=preserve_trailing,
            prompt_segments=[{"literal": "<audio>"}],
        ),
    )

    # Match the CLI order: neural graphs are saved before metadata generation
    # registers the workflow policy components.
    pkg.save(str(tmp_path), progress_bar=False, check_weights=False)
    artifacts = write_onnx_genai_config(pkg, str(tmp_path))
    with open(artifacts["inference_metadata"]) as handle:
        workflow = yaml.safe_load(handle)["pipeline"]["workflow"]

    assert workflow["outputs"]["audio"]["media"] == {
        "container": "wav",
        "encoding": "pcm_s16_le",
        "sample_rate_hz": target_rate,
        "source_sample_rate_hz": source_rate,
        "channels": channels,
        "delivery": "buffered",
    }
    assert len(workflow["serving"]["state_service"]["groups"]) == 1
    expanded = {"kind": "request_expanded", "axis": 0, "factor": 2}
    assert workflow["inputs"]["request.prompt_tokens"]["contract"]["batch_layout"] == expanded
    assert workflow["state"]["global_logits"]["contract"]["batch_layout"] == expanded
    assert workflow["state"]["global_hidden"]["contract"]["batch_layout"] == expanded
    assert workflow["state"]["frame_history"]["contract"]["shape"][-1] == (
        hidden_size * (residual_codebooks + 1)
    )
    assert workflow["state"]["flow_latents"]["contract"]["shape"][1] == latent_channels
    assert workflow["state"]["previous_condition"]["contract"]["shape"][-1] == condition_size
    assert workflow["inputs"]["package.local_steps"]["default"] == residual_codebooks
    assert workflow["inputs"]["package.flow_steps"]["default"] == flow_steps
    serialized = yaml.safe_dump(workflow["steps"])
    assert serialized.count("cell: global_cache_") == 1
    assert "local_cache" not in serialized
    assert (tmp_path / "speech_processor.json").is_file()
    with open(tmp_path / "speech_processor.json") as handle:
        processor = json.load(handle)
    assert processor["max_output_units"] == 9000
    assert processor["guidance_rows"]["unconditional_token_id"] == stop_token - 1
    assert processor["guidance_rows"]["replace_from"] == replace_from
    assert processor["guidance_rows"]["preserve_trailing"] == preserve_trailing
    if hidden_size != 8:
        full_metadata = yaml.safe_dump(workflow)
        assert "151675" not in full_metadata
        assert "16384" not in full_metadata
    policy_artifacts = {
        component["implementation"]["artifact"]
        for component in workflow["components"].values()
        if component["implementation"].get("kind") == "onnx"
        and component["implementation"]["artifact"].startswith("policies/")
    }
    assert policy_artifacts
    for artifact in policy_artifacts:
        policy_path = tmp_path / artifact
        assert policy_path.is_file(), artifact
        ir.load(policy_path)
    base_config = pkg.config.workflow_config
    policy_names = set(pkg.policy_components)
    # The metadata writer -- not the typed config -- is the authority on the two
    # control-token bounds, because the upper bounds depend on the built graph's
    # global logits width. Each invalid token is a well-formed config the writer
    # must still reject before writing any artifact or registering any policy.
    invalid_tokens = [
        ("negative-stop", "stop_token_id", -1),
        ("stop-at-semantic-start", "stop_token_id", semantic_start),
        ("stop-inside-semantic-range", "stop_token_id", semantic_start + 1),
        ("stop-above-logits", "stop_token_id", 170000),
        (
            "unconditional-inside-semantic-range",
            "unconditional_token_id",
            semantic_start + 1,
        ),
        ("negative-unconditional", "unconditional_token_id", -1),
        ("unconditional-above-logits", "unconditional_token_id", 170000),
    ]
    for label, field, invalid_value in invalid_tokens:
        pkg.config.workflow_config = dataclasses.replace(base_config, **{field: invalid_value})
        with pytest.raises(ValueError, match=field):
            build_hierarchical_audio_workflow_metadata(pkg)
        invalid_dir = tmp_path / label
        with pytest.raises(ValueError, match=field):
            write_onnx_genai_config(pkg, str(invalid_dir))
        assert not list(invalid_dir.rglob("*"))
        assert set(pkg.policy_components) == policy_names
    pkg.config.workflow_config = base_config

    # Completeness of the remaining fields fails closed at construction: the
    # typed config refuses to describe a workflow whose frame ceiling is absent.
    with pytest.raises(ValueError, match="max_audio_frames"):
        dataclasses.replace(base_config, max_audio_frames=0)

    # Fail closed on an unresolved config: a structurally-hierarchical package
    # (marked by ``workflow_kind``) whose workflow config could not be resolved
    # must still be routed to the hierarchical writer -- never misclassified as
    # diffusion -- and refuse to emit metadata with a targeted instruction,
    # leaving no artifacts or policy components behind.
    pkg.config.workflow_config = None
    pkg.config.workflow_kind = "hierarchical_audio"
    unresolved_dir = tmp_path / "unresolved-config"
    with pytest.raises(ValueError, match="build_diffusers_pipeline"):
        write_onnx_genai_config(pkg, str(unresolved_dir))
    assert not list(unresolved_dir.rglob("*"))
    assert set(pkg.policy_components) == policy_names


@dataclasses.dataclass
class _TTSSubCfg:
    num_code_groups: int = 16


@dataclasses.dataclass
class _TTSCfg(_Cfg):
    tts: _TTSSubCfg = dataclasses.field(default_factory=_TTSSubCfg)


class _TTSPkg(dict):
    config = _TTSCfg()


def test_dispatch_multi_decoder_tts_with_pre_embedder(tmp_path):
    pkg = ModelPackage(
        {
            "talker": _model(
                "talker",
                [_value("inputs_embeds", ir.DataType.FLOAT, ["batch", "sequence", 16])],
                [("last_hidden_state", ir.DataType.FLOAT, ["batch", 16])],
            ),
            "code_predictor": _model(
                "code_predictor",
                [
                    _value("last_hidden_state", ir.DataType.FLOAT, ["batch", 16]),
                    _value("step_index", ir.DataType.INT64, ["batch"]),
                ],
                [("logits", ir.DataType.FLOAT, ["batch", 64])],
            ),
            "talker_step_embedder": _model(
                "talker_step_embedder",
                [_value("frame_codes", ir.DataType.INT64, ["batch", 16])],
                [("inputs_embeds", ir.DataType.FLOAT, ["batch", 1, 16])],
            ),
            "talker_prefill_embedder": _model(
                "talker_prefill_embedder",
                [_value("text_ids", ir.DataType.INT64, ["batch", "sequence"])],
                [("prefill_embeds", ir.DataType.FLOAT, ["batch", "sequence", 16])],
            ),
            "codec": _model(
                "codec",
                [_value("codes", ir.DataType.INT64, ["batch", 16, "frames"])],
                [("waveform", ir.DataType.FLOAT, ["batch", 1, "samples"])],
            ),
        },
        config=_TTSCfg(),
    )
    artifacts = write_onnx_genai_config(pkg, str(tmp_path))
    with open(artifacts["inference_metadata"]) as handle:
        workflow = yaml.safe_load(handle)["pipeline"]["workflow"]
    outer = workflow["steps"][0]
    assert outer["iteration"]["value"] == "talker.iteration"
    assert outer["steps"][2]["iteration"]["value"] == "code.iteration"
    assert (tmp_path / "policies" / "code_frame_update.onnx").is_file()


def test_unrecognized_valid_multi_component_package_exports_contracts(tmp_path):
    pkg = _EncoderDecoderPkg(
        {
            "widget": _FakeModel(["x"], ["widget_output"]),
            "gadget": _FakeModel(["y"], ["gadget_output"]),
        }
    )
    artifacts = write_onnx_genai_config(pkg, str(tmp_path))

    compatibility = json.loads(
        Path(artifacts["runtime_compatibility"]).read_text(encoding="utf-8")
    )
    assert compatibility["runtime_validation_status"] == "unsupported-by-tested-runtime"
    assert compatibility["components"] == {
        "widget": {
            "filename": "widget/model.onnx",
            "inputs": ["x"],
            "outputs": ["widget_output"],
            "metadata": {},
        },
        "gadget": {
            "filename": "gadget/model.onnx",
            "inputs": ["y"],
            "outputs": ["gadget_output"],
            "metadata": {},
        },
    }


def test_unrecognized_configless_multi_component_package_exports_contracts(tmp_path):
    pkg = {
        "widget": _FakeModel(["x"], ["widget_output"]),
        "gadget": _FakeModel(["y"], ["gadget_output"]),
    }

    artifacts = write_onnx_genai_config(pkg, str(tmp_path))
    compatibility = json.loads(
        Path(artifacts["runtime_compatibility"]).read_text(encoding="utf-8")
    )

    assert compatibility["runtime_validation_status"] == "unsupported-by-tested-runtime"
    assert set(compatibility["components"]) == {"widget", "gadget"}


def test_unrecognized_malformed_component_contract_still_fails(tmp_path):
    pkg = _EncoderDecoderPkg(
        {
            "widget": _FakeModel(["x"], ["widget_output"]),
            "gadget": object(),
        }
    )

    with pytest.raises(ValueError, match=r"gadget.*no graph contract"):
        write_onnx_genai_config(pkg, str(tmp_path))


def test_decoder_emits_tokenizer_from_source(tmp_path):
    # A text-producing package emits tokenizer.json from its HF source so the
    # onnx-genai package is self-contained.
    from unittest import mock

    saved = {}

    class _FakeBackend:
        def save(self, path):
            saved["path"] = path
            with open(path, "w") as handle:
                handle.write("{}")

    fake_tok = mock.Mock()
    fake_tok.backend_tokenizer = _FakeBackend()
    fake_tf = mock.Mock()
    fake_tf.AutoTokenizer.from_pretrained.return_value = fake_tok

    with mock.patch.dict("sys.modules", {"transformers": fake_tf}):
        artifacts = write_onnx_genai_config(
            _decoder_package(), str(tmp_path), config=_Cfg(), source="some/model-id"
        )

    assert artifacts.get("tokenizer") == str(tmp_path / "tokenizer.json")
    assert (tmp_path / "tokenizer.json").exists()
    fake_tf.AutoTokenizer.from_pretrained.assert_called_once()


def test_decoder_without_source_skips_tokenizer(tmp_path):
    artifacts = write_onnx_genai_config(_decoder_package(), str(tmp_path), config=_Cfg())
    assert "tokenizer" not in artifacts
    assert not (tmp_path / "tokenizer.json").exists()


def test_decoder_package_ships_chat_template_assets(tmp_path):
    """A text decoder package must carry the assets needed to build a prompt.

    ``tokenizer.json`` alone leaves the runtime with no chat template, so an
    instruction-tuned decoder receives raw user text with no BOS and no turn
    markers. Image/audio processor configs, by contrast, describe media a text
    package cannot consume and must not be copied.
    """
    source = tmp_path / "source"
    source.mkdir()
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "image_processor.json",
    ):
        (source / filename).write_text("{}", encoding="utf-8")

    output = tmp_path / "output"
    write_onnx_genai_config(
        _decoder_package(_Int4Cfg()),
        str(output),
        config=_Int4Cfg(),
        source=str(source),
    )

    assert (output / "tokenizer.json").is_file()
    assert (output / "tokenizer_config.json").is_file()
    assert (output / "special_tokens_map.json").is_file()
    assert (output / "chat_template.jinja").is_file()
    assert not (output / "preprocessor_config.json").exists()
    assert not (output / "image_processor.json").exists()
