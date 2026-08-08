# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Complete compositional exporter for NVIDIA Cosmos3-Omni world models.

The public Cosmos3-Omni checkpoint combines several independently executable
neural systems:

* a Qwen3-VL Reasoner (decoder, vision encoder, and embedding graphs);
* the unified MoT rectified-flow transformer, including optional Sound and
  Action heads;
* a Wan video VAE (encoder and decoder);
* an optional Cosmos3 AVAE sound tokenizer (full or decoder-only).

This module builds those graphs through their normal Mobius tasks and composes
them into a :class:`mobius.PipelinePackage`. The topology manifest stays
runtime-agnostic: schedulers, tokenizers, and processors are copied as opaque
assets, while transform edges declare the capabilities a runtime must supply.
"""

from __future__ import annotations

__all__ = ["build_cosmos3_world_model"]

import dataclasses
import json
import logging
import pathlib
from collections.abc import Iterable, Mapping
from typing import Any

import onnx_ir as ir
import safetensors.torch
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors import safe_open

from mobius._builder import build as build_model
from mobius._builder import build_from_module, resolve_dtype
from mobius._configs import (
    Cosmos3AudioConfig,
    Cosmos3OmniGeneratorConfig,
    WanVAEConfig,
)
from mobius._diffusers_builder import _download_diffusers_component_weights
from mobius._model_package import ModelPackage
from mobius._pipeline import (
    PipelineBuilder,
    PipelinePackage,
    register_transform,
)
from mobius._weight_loading import _dequantize_fp8_weights, iter_weight_shards
from mobius.models.cosmos3_audio import create_cosmos3_avae_audio_tokenizer
from mobius.models.cosmos3_omni import Cosmos3OmniReasonerModel
from mobius.models.cosmos3_omni_generator import Cosmos3OmniGeneratorModel
from mobius.models.wan_vae import AutoencoderKLWanModel
from mobius.tasks._cosmos3_audio import select_cosmos3_audio_task
from mobius.tasks._wan_vae import WanVAETask

logger = logging.getLogger(__name__)

_VIDEO_DIFFUSION_FINALIZE = "video_diffusion_finalize"
register_transform(
    _VIDEO_DIFFUSION_FINALIZE,
    description=(
        "Advance the final scheduler step and restore packed diffusion tokens "
        "to a video decoder's latent tensor layout"
    ),
    capabilities=(
        "iterative_scheduler",
        "tensor_patchify",
        "tensor_reshape",
        "tensor_cast",
    ),
    required_parameters=(
        "scheduler_asset",
        "state",
        "spatial_patch_size",
        "latent_channels",
        "input_layout",
        "output_layout",
        "source_dtype",
        "target_dtype",
    ),
    allowed_parameters=(
        "scheduler_asset",
        "state",
        "spatial_patch_size",
        "latent_channels",
        "input_layout",
        "output_layout",
        "source_dtype",
        "target_dtype",
    ),
)
_AUDIO_DIFFUSION_FINALIZE = "audio_diffusion_finalize"
register_transform(
    _AUDIO_DIFFUSION_FINALIZE,
    description=(
        "Advance the final scheduler step and restore packed sound tokens "
        "to an audio decoder's latent tensor layout"
    ),
    capabilities=(
        "iterative_scheduler",
        "tensor_reshape",
        "tensor_cast",
    ),
    required_parameters=(
        "scheduler_asset",
        "state",
        "input_layout",
        "output_layout",
        "source_dtype",
        "target_dtype",
    ),
    allowed_parameters=(
        "scheduler_asset",
        "state",
        "input_layout",
        "output_layout",
        "source_dtype",
        "target_dtype",
    ),
)

_REASONER_NAMES = {
    "decoder": "reasoner_decoder",
    "vision_encoder": "reasoner_vision_encoder",
    "embedding": "reasoner_embedding",
}
_GENERATOR_NAME = "generator"
_VIDEO_ENCODER_NAME = "video_encoder"
_VIDEO_DECODER_NAME = "video_decoder"
_AUDIO_ENCODER_NAME = "audio_encoder"
_AUDIO_DECODER_NAME = "audio_decoder"

# Public diffusers Cosmos3 action-domain contract. Domain IDs select the
# DomainAwareLinear bank; raw widths remove zero-padding from action outputs.
_ACTION_DOMAIN_IDS: dict[str, int] = {
    "no_action": 0,
    "av": 1,
    "camera_pose": 2,
    "hand_pose": 3,
    "pusht": 4,
    "libero": 5,
    "umi": 6,
    "bridge_orig_lerobot": 7,
    "droid_lerobot": 8,
    "robomind-franka": 8,
    "galbot": 9,
    "robomind-franka-dual": 12,
    "robomind-ur": 13,
    "agibotworld": 15,
    "agibot_gear_gripper": 15,
    "agibot_gear_gripper_ext": 15,
    "fractal": 20,
}
_ACTION_RAW_DIMS: dict[str, int] = {
    "no_action": 0,
    "av": 9,
    "camera_pose": 9,
    "hand_pose": 57,
    "pusht": 2,
    "umi": 10,
    "bridge_orig_lerobot": 10,
    "droid_lerobot": 10,
    "robomind-franka": 10,
    "galbot": 30,
    "robomind-franka-dual": 20,
    "robomind-ur": 10,
    "agibotworld": 29,
    "agibot_gear_gripper": 29,
    "agibot_gear_gripper_ext": 29,
    "fractal": 10,
}

_ASSET_CANDIDATES: tuple[tuple[str, bool], ...] = (
    ("config.json", True),
    ("model_index.json", True),
    ("transformer/config.json", True),
    ("vae/config.json", True),
    ("vision_encoder/config.json", False),
    ("scheduler/scheduler_config.json", True),
    ("tokenizer.json", False),
    ("tokenizer_config.json", False),
    ("special_tokens_map.json", False),
    ("vocab.json", False),
    ("merges.txt", False),
    ("chat_template.json", False),
    ("chat_template.jinja", False),
    ("generation_config.json", False),
    ("checkpoint.json", False),
    ("preprocessor_config.json", False),
    ("video_preprocessor_config.json", False),
    ("processor_config.json", False),
    ("text_tokenizer/tokenizer.json", False),
    ("text_tokenizer/tokenizer_config.json", False),
    ("text_tokenizer/vocab.json", False),
    ("text_tokenizer/merges.txt", False),
)


def _resolve_file(model_id: str, filename: str, *, required: bool = True) -> str | None:
    """Resolve one local-or-Hub checkpoint file without interpreting it."""
    root = pathlib.Path(model_id)
    if root.is_dir():
        path = (root / pathlib.PurePosixPath(filename)).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(
                f"Checkpoint file escapes model directory: {filename!r}"
            ) from error
        if path.is_file():
            return str(path)
        if required:
            raise FileNotFoundError(f"Required checkpoint file not found: {path}")
        return None
    try:
        return hf_hub_download(repo_id=model_id, filename=filename)
    except EntryNotFoundError:
        if required:
            raise
        return None


def _load_json(model_id: str, filename: str) -> tuple[dict[str, Any], str]:
    path = _resolve_file(model_id, filename)
    assert path is not None
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{filename!r} must contain a JSON object")
    return value, path


def _load_optional_json(model_id: str, filename: str) -> dict[str, Any]:
    path = _resolve_file(model_id, filename, required=False)
    if path is None:
        return {}
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{filename!r} must contain a JSON object")
    return value


def _component_class(
    pipeline_index: Mapping[str, Any],
    component: str,
) -> str | None:
    info = pipeline_index.get(component)
    if info in (None, [None, None]):
        return None
    if not isinstance(info, list) or len(info) != 2 or not isinstance(info[1], str):
        raise ValueError(f"Invalid model_index.json entry for {component!r}: {info!r}")
    return info[1]


def _component_weight_names(model_id: str, component: str) -> set[str]:
    """Read only safetensors metadata to determine component graph shape."""
    index_names = (
        "diffusion_pytorch_model.safetensors.index.json",
        "model.safetensors.index.json",
    )
    single_names = (
        "diffusion_pytorch_model.safetensors",
        "model.safetensors",
    )
    root = pathlib.Path(model_id)
    if root.is_dir():
        component_dir = root / component
        for name in index_names:
            path = component_dir / name
            if path.is_file():
                with path.open(encoding="utf-8") as handle:
                    index = json.load(handle)
                return set(index["weight_map"])
        for name in single_names:
            path = component_dir / name
            if path.is_file():
                with safe_open(str(path), framework="pt", device="cpu") as file:
                    return set(file.keys())
        raise FileNotFoundError(
            f"No safetensors checkpoint found for component {component!r} in {model_id!r}"
        )

    for name in index_names:
        filename = f"{component}/{name}"
        path = _resolve_file(model_id, filename, required=False)
        if path is not None:
            with open(path, encoding="utf-8") as handle:
                index = json.load(handle)
            return set(index["weight_map"])
    api = HfApi()
    for name in single_names:
        filename = f"{component}/{name}"
        try:
            metadata = api.parse_safetensors_file_metadata(model_id, filename)
        except EntryNotFoundError:
            continue
        return set(metadata.tensors)
    raise FileNotFoundError(
        f"No safetensors checkpoint found for component {component!r} in {model_id!r}"
    )


def _safe_component_files(model_dir: pathlib.Path, component: str) -> list[pathlib.Path]:
    """Resolve local component shards with traversal protection."""
    component_dir = (model_dir / component).resolve()
    try:
        component_dir.relative_to(model_dir.resolve())
    except ValueError as error:
        raise ValueError(f"Unsafe component path {component!r}") from error

    for basename in (
        "diffusion_pytorch_model",
        "model",
    ):
        index_path = component_dir / f"{basename}.safetensors.index.json"
        if not index_path.is_file():
            continue
        with index_path.open(encoding="utf-8") as handle:
            index = json.load(handle)
        paths: list[pathlib.Path] = []
        for filename in sorted(set(index["weight_map"].values())):
            relative = pathlib.PurePosixPath(str(filename).replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe component weight filename: {filename!r}")
            path = (component_dir / relative).resolve()
            try:
                path.relative_to(component_dir)
            except ValueError as error:
                raise ValueError(
                    f"Component weight filename escapes its directory: {filename!r}"
                ) from error
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(path)
        return paths

    for basename in ("diffusion_pytorch_model.safetensors", "model.safetensors"):
        path = component_dir / basename
        if path.is_file():
            return [path]
    raise FileNotFoundError(
        f"No safetensors checkpoint found for component {component!r} in {model_dir}"
    )


def _load_component_weights(
    model_id: str,
    component: str,
) -> dict[str, Any]:
    """Load one diffusers component, supporting Hub and local directories."""
    root = pathlib.Path(model_id)
    if not root.is_dir():
        return _download_diffusers_component_weights(model_id, component)

    state_dict: dict[str, Any] = {}
    for path in _safe_component_files(root, component):
        state_dict.update(safetensors.torch.load_file(str(path)))
    return _dequantize_fp8_weights(state_dict)


def _resolved_dtype(dtype: str | ir.DataType | None) -> ir.DataType | None:
    if dtype is None or isinstance(dtype, ir.DataType):
        return dtype
    return resolve_dtype(dtype)


def _build_components(
    model_id: str,
    *,
    dtype: str | ir.DataType | None,
    execution_provider: str,
    trace_optimization: bool,
    pipeline_index: Mapping[str, Any],
    transformer_config_dict: Mapping[str, Any],
    vae_config_dict: Mapping[str, Any],
    audio_config_dict: Mapping[str, Any] | None,
    audio_weight_names: Iterable[str] | None,
    has_reasoner_vision: bool,
    reasoner_module_class: type[Any] = Cosmos3OmniReasonerModel,
    reasoner_task: str | None = None,
) -> tuple[
    ModelPackage,
    Any,
    ModelPackage,
    Cosmos3OmniGeneratorModel,
    ModelPackage,
    AutoencoderKLWanModel,
    ModelPackage | None,
    Any | None,
]:
    """Build all component graphs while retaining modules for weight routing."""
    reasoner_package = build_model(
        model_id,
        task=reasoner_task,
        module_class=reasoner_module_class,
        dtype=dtype,
        load_weights=False,
        execution_provider=execution_provider,
        trace_optimization=trace_optimization,
    )
    if not has_reasoner_vision:
        reasoner_package.pop("vision_encoder", None)
    expected_reasoner = {"decoder", "embedding"}
    if has_reasoner_vision:
        expected_reasoner.add("vision_encoder")
    if set(reasoner_package) != expected_reasoner:
        raise ValueError(
            "Cosmos3 Reasoner components do not match the checkpoint layout; "
            f"got {sorted(reasoner_package)}"
        )
    reasoner_module = reasoner_module_class(reasoner_package.config)

    generator_config = Cosmos3OmniGeneratorConfig.from_diffusers(transformer_config_dict)
    if resolved_dtype := _resolved_dtype(dtype):
        generator_config = dataclasses.replace(generator_config, dtype=resolved_dtype)
        generator_config.validate()
    generator_module = Cosmos3OmniGeneratorModel(generator_config)
    generator_package = build_from_module(
        generator_module,
        generator_config,
        task="cosmos3-omni-generator",
        execution_provider=execution_provider,
        trace_optimization=trace_optimization,
    )

    vae_config = WanVAEConfig.from_diffusers(vae_config_dict)
    vae_module = AutoencoderKLWanModel(vae_config)
    vae_package = build_from_module(
        vae_module,
        vae_config,
        task=WanVAETask(),
        execution_provider=execution_provider,
        trace_optimization=trace_optimization,
    )

    audio_package: ModelPackage | None = None
    audio_module: Any | None = None
    sound_class = _component_class(pipeline_index, "sound_tokenizer")
    if generator_config.sound_gen and sound_class is None:
        raise ValueError(
            "The transformer enables Sound generation, but model_index.json has no "
            "sound_tokenizer component."
        )
    if sound_class is not None:
        if sound_class != "Cosmos3AVAEAudioTokenizer":
            raise ValueError(f"Unsupported Cosmos3 sound tokenizer class {sound_class!r}")
        if audio_config_dict is None or audio_weight_names is None:
            raise ValueError("Sound tokenizer config and weight metadata are required")
        audio_config = Cosmos3AudioConfig.from_diffusers(
            audio_config_dict,
            weight_names=audio_weight_names,
        )
        audio_module = create_cosmos3_avae_audio_tokenizer(audio_config)
        audio_task = select_cosmos3_audio_task(audio_config)()
        audio_package = build_from_module(
            audio_module,
            audio_config,
            task=audio_task,
            execution_provider=execution_provider,
            trace_optimization=trace_optimization,
        )

    return (
        reasoner_package,
        reasoner_module,
        generator_package,
        generator_module,
        vae_package,
        vae_module,
        audio_package,
        audio_module,
    )


def _apply_checkpoint_weights(
    model_id: str,
    *,
    reasoner_package: ModelPackage,
    reasoner_module: Any,
    generator_package: ModelPackage,
    generator_module: Cosmos3OmniGeneratorModel,
    vae_package: ModelPackage,
    vae_module: AutoencoderKLWanModel,
    audio_package: ModelPackage | None,
    audio_module: Any | None,
) -> None:
    """Stream the shared checkpoint once, routing each shard to both towers."""
    if reasoner_module.config.tie_word_embeddings:
        raise ValueError(
            "Streaming a tied-embedding Cosmos3 Reasoner is not supported because "
            "the shared embedding and LM-head tensors may live in different shards."
        )
    generator_seen: set[str] = set()
    for shard in iter_weight_shards(model_id):
        reasoner_package.apply_weights_partial(reasoner_module.preprocess_weights(shard))
        generator_weights = generator_module.preprocess_weight_shard(shard)
        generator_seen.update(generator_weights)
        generator_package.apply_weights_partial(generator_weights)

    missing_generator = sorted(generator_module.expected_checkpoint_keys() - generator_seen)
    if missing_generator:
        raise ValueError(
            "Cosmos3 unified checkpoint is missing Generator weights required by the "
            f"configured architecture: {missing_generator[:16]}"
        )
    reasoner_package.finalize_weights()
    generator_package.finalize_weights()

    vae_weights = _load_component_weights(model_id, "vae")
    vae_package.apply_weights(vae_module.preprocess_weights(vae_weights))

    if audio_package is not None and audio_module is not None:
        audio_weights = _load_component_weights(model_id, "sound_tokenizer")
        audio_package.apply_weights(audio_module.preprocess_weights(audio_weights))

    reasoner_package.validate_weights()
    generator_package.validate_weights()
    vae_package.validate_weights()
    if audio_package is not None:
        audio_package.validate_weights()


def _connect_same_named_ports(
    builder: PipelineBuilder,
    source_name: str,
    source_model: ir.Model,
    target_name: str,
    target_model: ir.Model,
    connected_targets: set[str],
) -> None:
    outputs = {value.name for value in source_model.graph.outputs}
    for value in target_model.graph.inputs:
        if value.name in outputs:
            connection = builder.connect(
                f"{source_name}.{value.name}",
                f"{target_name}.{value.name}",
            )
            connected_targets.add(connection.target.qualified)


def _component_metadata(dtype: ir.DataType, **values: Any) -> dict[str, Any]:
    return {"dtype": dtype.name, **values}


def _preferred_execution_providers(
    requested: str,
    dtype: ir.DataType,
) -> tuple[str, ...]:
    if requested != "default":
        return (requested,)
    if dtype == ir.DataType.BFLOAT16:
        return ("cuda", "cpu")
    return ("cuda", "dml", "cpu")


def _compose_pipeline(
    *,
    model_id: str,
    reasoner_package: ModelPackage,
    generator_package: ModelPackage,
    vae_package: ModelPackage,
    audio_package: ModelPackage | None,
    generator_config: Cosmos3OmniGeneratorConfig,
    vae_config: WanVAEConfig,
    scheduler_config: Mapping[str, Any],
    assets: Mapping[str, tuple[str, bool]],
    pipeline_model_type: str = "cosmos3_omni",
    reasoner_architecture: str = "qwen3_vl",
    extra_metadata: Mapping[str, Any] | None = None,
    execution_provider: str = "default",
    generation_config: Mapping[str, Any] | None = None,
    default_inference_steps: int = 35,
    default_action_domain: str = "no_action",
    scheduler_mode_overrides: Mapping[str, Any] | None = None,
) -> PipelinePackage:
    """Compose already-built component graphs into the complete topology."""
    if generator_config.sound_gen and audio_package is None:
        raise ValueError(
            "A transformer with sound_gen=True requires a sound-tokenizer package."
        )
    if vae_config.z_dim != generator_config.latent_channel:
        raise ValueError(
            "Cosmos3 video VAE latent width must match the Generator vision head: "
            f"{vae_config.z_dim} != {generator_config.latent_channel}."
        )
    if audio_package is not None:
        audio_config = audio_package.config
        if audio_config.vocoder_input_dim != generator_config.sound_dim:
            raise ValueError(
                "Cosmos3 AVAE latent width must match the Generator Sound head: "
                f"{audio_config.vocoder_input_dim} != {generator_config.sound_dim}."
            )
    if generator_config.action_gen:
        assert generator_config.action_dim is not None
        if max(_ACTION_DOMAIN_IDS.values()) >= generator_config.num_embodiment_domains:
            raise ValueError("Cosmos3 action domain metadata exceeds num_embodiment_domains.")
        oversized = {
            name: width
            for name, width in _ACTION_RAW_DIMS.items()
            if width > generator_config.action_dim
        }
        if oversized:
            raise ValueError(f"Cosmos3 raw action widths exceed action_dim: {oversized}.")
        if default_action_domain not in _ACTION_DOMAIN_IDS:
            raise ValueError(
                f"Unknown default Cosmos3 action domain {default_action_domain!r}."
            )
    builder = PipelineBuilder()
    component_configs: dict[str, object] = {}
    models: dict[str, ir.Model] = {}

    for key, name in _REASONER_NAMES.items():
        if key not in reasoner_package:
            continue
        model = reasoner_package[key]
        role = {"decoder": "decoder", "vision_encoder": "encoder", "embedding": "embedding"}[
            key
        ]
        builder.add_model(
            name,
            model,
            role=role,
            source=model_id,
            config=_component_metadata(
                reasoner_package.config.dtype,
                subsystem="reasoner",
                architecture=reasoner_architecture,
            ),
            preferred_execution_providers=_preferred_execution_providers(
                execution_provider,
                reasoner_package.config.dtype,
            ),
            parameter_dtype=reasoner_package.config.dtype.name,
        )
        models[name] = model
        component_configs[name] = reasoner_package.config

    generator = generator_package["model"]
    builder.add_model(
        _GENERATOR_NAME,
        generator,
        role="dynamics",
        source=model_id,
        config=_component_metadata(
            generator_config.dtype,
            subsystem="generator",
            sound_gen=generator_config.sound_gen,
            action_gen=generator_config.action_gen,
            patch_latent_dim=generator_config.patch_latent_dim,
        ),
        preferred_execution_providers=_preferred_execution_providers(
            execution_provider,
            generator_config.dtype,
        ),
        parameter_dtype=generator_config.dtype.name,
    )
    models[_GENERATOR_NAME] = generator
    component_configs[_GENERATOR_NAME] = generator_config

    video_names = {"encoder": _VIDEO_ENCODER_NAME, "decoder": _VIDEO_DECODER_NAME}
    for key, name in video_names.items():
        model = vae_package[key]
        builder.add_model(
            name,
            model,
            role=key,
            presence="video_conditioning" if key == "encoder" else None,
            source=model_id,
            config=_component_metadata(
                vae_config.dtype,
                subsystem="video_vae",
                spatial_compression=vae_config.scale_factor_spatial,
                temporal_compression=vae_config.scale_factor_temporal,
            ),
            preferred_execution_providers=_preferred_execution_providers(
                execution_provider,
                vae_config.dtype,
            ),
            parameter_dtype=vae_config.dtype.name,
        )
        models[name] = model
        component_configs[name] = vae_config

    audio_config: Cosmos3AudioConfig | None = None
    if audio_package is not None:
        audio_config = audio_package.config
        audio_names = {"decoder": _AUDIO_DECODER_NAME}
        if "encoder" in audio_package:
            audio_names["encoder"] = _AUDIO_ENCODER_NAME
        for key, name in audio_names.items():
            model = audio_package[key]
            builder.add_model(
                name,
                model,
                role=key,
                presence="audio_conditioning" if key == "encoder" else None,
                source=model_id,
                config=_component_metadata(
                    audio_config.dtype,
                    subsystem="sound_tokenizer",
                    sample_rate=audio_config.sampling_rate,
                    hop_size=audio_config.resolved_hop_size,
                ),
                preferred_execution_providers=_preferred_execution_providers(
                    execution_provider,
                    audio_config.dtype,
                ),
                parameter_dtype=audio_config.dtype.name,
            )
            models[name] = model
            component_configs[name] = audio_config

    initial_targets: set[str] = set()
    recurrent_targets: set[str] = set()
    if _REASONER_NAMES["vision_encoder"] in models:
        _connect_same_named_ports(
            builder,
            _REASONER_NAMES["vision_encoder"],
            models[_REASONER_NAMES["vision_encoder"]],
            _REASONER_NAMES["embedding"],
            models[_REASONER_NAMES["embedding"]],
            initial_targets,
        )
    _connect_same_named_ports(
        builder,
        _REASONER_NAMES["embedding"],
        models[_REASONER_NAMES["embedding"]],
        _REASONER_NAMES["decoder"],
        models[_REASONER_NAMES["decoder"]],
        initial_targets,
    )

    reasoner_decoder = models[_REASONER_NAMES["decoder"]]
    decoder_inputs = {value.name for value in reasoner_decoder.graph.inputs}
    state_specs: list[dict[str, Any]] = []
    for output in reasoner_decoder.graph.outputs:
        if not output.name.startswith("present."):
            continue
        cache_input = f"past_key_values.{output.name.removeprefix('present.')}"
        if cache_input in decoder_inputs:
            connection = builder.connect(
                f"{_REASONER_NAMES['decoder']}.{output.name}",
                f"{_REASONER_NAMES['decoder']}.{cache_input}",
                recurrent=True,
            )
            recurrent_targets.add(connection.target.qualified)
            cache_suffix = output.name.removeprefix("present.").replace(".", "_")
            state_specs.append(
                {
                    "name": f"reasoner_kv_{cache_suffix}",
                    "kind": "kv_cache",
                    "input": connection.target,
                    "output": connection.source,
                    "lifetime": "sequence",
                    "release_after": "reasoner_decode",
                    "sequence_axis": 2,
                    "metadata": {
                        "update": "append",
                        "initializer": "empty_tensor",
                    },
                }
            )

    iterative_state_inputs = ["vision_tokens"]
    for output_name, input_name in (
        ("vision_pred", "vision_tokens"),
        ("sound_pred", "sound_tokens"),
        ("action_pred", "action_tokens"),
    ):
        if output_name not in {value.name for value in generator.graph.outputs}:
            continue
        state_name = input_name.removesuffix("_tokens") + "_state"
        connection = builder.connect(
            f"{_GENERATOR_NAME}.{output_name}",
            f"{_GENERATOR_NAME}.{input_name}",
            recurrent=True,
            transform="scheduler_step",
            parameters={
                "scheduler_asset": "scheduler/scheduler_config.json",
                "stage": "world_generation",
                "state": state_name,
                "timestep_input": f"{_GENERATOR_NAME}.{input_name.removesuffix('_tokens')}_timesteps",
            },
        )
        recurrent_targets.add(connection.target.qualified)
        state_kind = "action_state" if input_name == "action_tokens" else "diffusion_latent"
        release_after = {
            "vision_tokens": "decode_video",
            "sound_tokens": "decode_audio",
            "action_tokens": "world_generation",
        }[input_name]
        state_specs.append(
            {
                "name": state_name,
                "kind": state_kind,
                "input": connection.target,
                "output": connection.source,
                "lifetime": "request",
                "release_after": release_after,
                "sequence_axis": 0,
                "metadata": {
                    "update": "scheduler_step",
                    "scheduler_asset": "scheduler/scheduler_config.json",
                },
            }
        )
        if input_name != "vision_tokens":
            iterative_state_inputs.append(input_name)

    video_final = builder.connect(
        f"{_GENERATOR_NAME}.vision_pred",
        f"{_VIDEO_DECODER_NAME}.latent",
        transform=_VIDEO_DIFFUSION_FINALIZE,
        context=(f"{_GENERATOR_NAME}.vision_tokens",),
        parameters={
            "scheduler_asset": "scheduler/scheduler_config.json",
            "state": "vision_state",
            "spatial_patch_size": generator_config.latent_patch_size,
            "latent_channels": generator_config.latent_channel,
            "input_layout": "packed_tokens",
            "output_layout": "BCTHW",
            "source_dtype": generator_config.dtype.name,
            "target_dtype": vae_config.dtype.name,
        },
    )
    initial_targets.add(video_final.target.qualified)

    if audio_package is not None:
        if "sound_pred" not in {value.name for value in generator.graph.outputs}:
            raise ValueError(
                "A sound tokenizer is present, but the transformer has no Sound output."
            )
        audio_final = builder.connect(
            f"{_GENERATOR_NAME}.sound_pred",
            f"{_AUDIO_DECODER_NAME}.latents",
            transform=_AUDIO_DIFFUSION_FINALIZE,
            context=(f"{_GENERATOR_NAME}.sound_tokens",),
            parameters={
                "scheduler_asset": "scheduler/scheduler_config.json",
                "state": "sound_state",
                "input_layout": "TC",
                "output_layout": "BCT",
                "source_dtype": generator_config.dtype.name,
                "target_dtype": audio_config.dtype.name,
            },
        )
        initial_targets.add(audio_final.target.qualified)

    generated_names = {
        "attention_mask",
        "position_ids",
        "cache_position",
        "text_indexes",
        "und_len",
        "vision_sequence_indexes",
        "vision_timesteps",
        "vision_timestep_token_indexes",
        "vision_mse_loss_indexes",
        "sound_sequence_indexes",
        "sound_timesteps",
        "sound_timestep_token_indexes",
        "sound_mse_loss_indexes",
        "action_domain_ids",
        "action_sequence_indexes",
        "action_timesteps",
        "action_timestep_token_indexes",
        "action_mse_loss_indexes",
        "action_pred_domain_ids",
    }

    def generated_program(
        component_name: str,
        input_name: str,
    ) -> tuple[str, dict[str, Any], str]:
        if input_name == "attention_mask":
            return (
                "causal_attention_mask",
                {
                    "sequence_input": f"{component_name}.inputs_embeds",
                    "past_state": [
                        state["name"] for state in state_specs if state["kind"] == "kv_cache"
                    ],
                    "visible_value": 1,
                    "masked_value": 0,
                },
                "attention.causal_mask",
            )
        if input_name == "position_ids":
            source = (
                f"{_GENERATOR_NAME}.input_ids"
                if component_name == _GENERATOR_NAME
                else f"{_REASONER_NAMES['embedding']}.input_ids"
            )
            sections = (
                generator_config.rope_axes_dim
                if component_name == _GENERATOR_NAME
                else (
                    getattr(reasoner_package.config, "mrope_section", None)
                    or generator_config.rope_axes_dim
                )
            )
            parameters: dict[str, Any] = {
                "source": source,
                "axes": 3,
                "mrope_sections": list(sections),
            }
            if component_name == _GENERATOR_NAME:
                parameters.update(
                    {
                        "temporal_margin": (
                            generator_config.unified_3d_mrope_temporal_modality_margin
                        ),
                        "reset_spatial": (generator_config.unified_3d_mrope_reset_spatial_ids),
                    }
                )
            else:
                parameters["past_state"] = [
                    state["name"] for state in state_specs if state["kind"] == "kv_cache"
                ]
            return ("multimodal_position_ids", parameters, "position.multimodal")
        if input_name.endswith("_timesteps"):
            modality = input_name.removesuffix("_timesteps")
            return (
                "scheduler_timesteps",
                {"stage": "world_generation", "modality": modality},
                f"diffusion.{modality}.timesteps",
            )
        if input_name in {"action_domain_ids", "action_pred_domain_ids"}:
            return (
                "action_domain_ids",
                {
                    "domain_input": "action_domain",
                    "default": default_action_domain,
                    "domain_map": _ACTION_DOMAIN_IDS,
                    "padded_dimension": generator_config.action_dim,
                },
                "action.domain_ids",
            )
        modality = input_name.split("_", 1)[0]
        source = (
            f"{_GENERATOR_NAME}.input_ids"
            if modality in {"text", "und"}
            else f"{_GENERATOR_NAME}.{modality}_tokens"
        )
        return (
            "packed_sequence_layout",
            {
                "modality": modality,
                "source": source,
                "layout": "flat_token_rows",
                "understanding_prefix": input_name == "und_len",
                "index_kind": input_name,
            },
            f"packing.{input_name}",
        )

    def external_semantic(component_name: str, input_name: str) -> str:
        if input_name == "input_ids":
            return "text.token_ids"
        if input_name == "pixel_values":
            return "vision.pixel_values"
        if input_name == "image_features":
            return "vision.image_features"
        if input_name == "audio":
            return "audio.waveform"
        if input_name == "sample":
            return "video.frames"
        if input_name.endswith("_tokens") and component_name == _GENERATOR_NAME:
            return f"diffusion.initial_{input_name.removesuffix('_tokens')}_latent"
        return f"tensor.{component_name}.{input_name}"

    for component_name, model in models.items():
        for value in model.graph.inputs:
            endpoint = f"{component_name}.{value.name}"
            if endpoint in initial_targets:
                continue
            if endpoint in recurrent_targets:
                if component_name == _GENERATOR_NAME:
                    alias = f"initial_{value.name}".replace(".", "_")
                    builder.declare_external(
                        endpoint,
                        alias=alias,
                        semantic=external_semantic(component_name, value.name),
                        required=True,
                    )
                else:
                    cache_axis = value.shape[2]
                    cache_axis_name = getattr(cache_axis, "value", None)
                    if not isinstance(cache_axis_name, str):
                        raise ValueError(
                            f"KV-cache input {endpoint!r} must expose a symbolic "
                            "sequence axis at index 2."
                        )
                    builder.declare_generated(
                        endpoint,
                        generator="empty_tensor",
                        parameters={
                            "dynamic_axes": {cache_axis_name: 0},
                            "fill": 0,
                        },
                        semantic=(
                            "kv_cache.key" if value.name.endswith(".key") else "kv_cache.value"
                        ),
                    )
                continue
            if value.name.startswith(("past_key_values.", "key_cache.", "value_cache.")):
                builder.declare_stateful(endpoint, semantic=f"state.{value.name}")
            elif value.name in generated_names:
                generator, parameters, semantic = generated_program(
                    component_name,
                    value.name,
                )
                builder.declare_generated(
                    endpoint,
                    generator=generator,
                    parameters=parameters,
                    semantic=semantic,
                )
            else:
                alias = f"{component_name}_{value.name}".replace(".", "_")
                presence = None
                required = True
                if component_name == _VIDEO_ENCODER_NAME:
                    presence = "video_conditioning"
                    required = False
                elif component_name == _AUDIO_ENCODER_NAME:
                    presence = "audio_conditioning"
                    required = False
                elif component_name == _REASONER_NAMES["vision_encoder"]:
                    presence = None
                    required = True
                builder.declare_external(
                    endpoint,
                    alias=alias,
                    semantic=external_semantic(component_name, value.name),
                    required=required,
                    presence=presence,
                )

    generation_values = dict(generation_config or {})
    eos_token_ids = generation_values.get("eos_token_id", [])
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    sampling = {
        "do_sample": generation_values.get("do_sample", False),
        "temperature": generation_values.get("temperature", 1.0),
        "top_k": generation_values.get("top_k", 50),
        "top_p": generation_values.get("top_p", 1.0),
        "repetition_penalty": generation_values.get("repetition_penalty", 1.0),
    }

    reasoner_prompt_components = [_REASONER_NAMES["embedding"]]
    if _REASONER_NAMES["vision_encoder"] in models:
        reasoner_prompt_components.insert(0, _REASONER_NAMES["vision_encoder"])
    builder.add_stage(
        "reasoner_prompt",
        "single_pass",
        reasoner_prompt_components,
        run_on="prefill",
    )
    builder.add_stage(
        "reasoner_decode",
        "autoregressive",
        [_REASONER_NAMES["embedding"], _REASONER_NAMES["decoder"]],
        run_on="decode",
        options={
            "tokenizer_asset": (
                "tokenizer.json"
                if "tokenizer.json" in assets
                else "text_tokenizer/tokenizer.json"
            ),
            "sampling": sampling,
            "stop": {
                "kind": "token_ids",
                "eos_token_ids": eos_token_ids,
                "max_sequence_length": getattr(
                    reasoner_package.config,
                    "max_position_embeddings",
                    None,
                ),
            },
            "max_tokens": {
                "default": generation_values.get("max_new_tokens"),
                "required_override": generation_values.get("max_new_tokens") is None,
                "limit": getattr(
                    reasoner_package.config,
                    "max_position_embeddings",
                    None,
                ),
            },
            "state_names": [
                state["name"] for state in state_specs if state["kind"] == "kv_cache"
            ],
        },
    )
    builder.add_stage(
        "world_generation",
        "iterative",
        [_GENERATOR_NAME],
        run_on="step",
        options={
            "scheduler": {
                "kind": scheduler_config.get("_class_name"),
                "config_asset": "scheduler/scheduler_config.json",
                "overrideable": [
                    "num_inference_steps",
                    "guidance_scale",
                    "flow_shift",
                    "use_karras_sigmas",
                ],
                "mode_overrides": dict(scheduler_mode_overrides or {}),
            },
            "default_steps": default_inference_steps,
            "timestep": {
                "generator": "scheduler_timesteps",
                "scale": generator_config.timestep_scale,
            },
            "prediction_type": scheduler_config.get("prediction_type", "flow_prediction"),
            "state_inputs": [f"{_GENERATOR_NAME}.{name}" for name in iterative_state_inputs],
            "packed_modalities": True,
        },
    )
    builder.add_stage(
        "encode_video",
        "on_demand",
        [_VIDEO_ENCODER_NAME],
        run_on="on_demand",
        options={"presence": "video_conditioning"},
    )
    builder.add_stage(
        "decode_video",
        "single_pass",
        [_VIDEO_DECODER_NAME],
        run_on="finalize",
    )

    if audio_package is not None:
        if "encoder" in audio_package:
            builder.add_stage(
                "encode_audio",
                "on_demand",
                [_AUDIO_ENCODER_NAME],
                run_on="on_demand",
                options={"presence": "audio_conditioning"},
            )
        builder.add_stage(
            "decode_audio",
            "single_pass",
            [_AUDIO_DECODER_NAME],
            run_on="finalize",
        )

    for state in state_specs:
        builder.add_state(**state)

    builder.add_public_output(f"{_REASONER_NAMES['decoder']}.logits", alias="logits")
    builder.add_public_output(f"{_VIDEO_DECODER_NAME}.sample", alias="video")
    builder.add_public_output(f"{_GENERATOR_NAME}.vision_pred", alias="vision_velocity")
    builder.add_public_output(f"{_VIDEO_ENCODER_NAME}.latent", alias="encoded_video_latent")
    if generator_config.action_gen:
        builder.add_public_output(
            f"{_GENERATOR_NAME}.action_pred",
            alias="action_velocity",
        )
        builder.add_public_state_output("action_state", alias="action")
    if audio_package is not None:
        builder.add_public_output(f"{_AUDIO_DECODER_NAME}.waveform", alias="sound")
        if "encoder" in audio_package:
            builder.add_public_output(
                f"{_AUDIO_ENCODER_NAME}.latent_mean",
                alias="encoded_audio_latent",
            )

    for destination, (source, required) in assets.items():
        builder.add_asset(destination, source, required=required)
    builder.set_profile(pipeline_model_type.replace("_", "-"), "1.0")
    builder.set_metadata("profile", "world-model")
    builder.set_metadata("model_type", pipeline_model_type)
    builder.set_metadata("source", model_id)
    builder.set_metadata(
        "modalities",
        {
            "vision": True,
            "sound": generator_config.sound_gen,
            "action": generator_config.action_gen,
        },
    )
    builder.set_metadata(
        "packing",
        {
            "generator_boundary": "packed_tokens",
            "latent_patch_size": generator_config.latent_patch_size,
            "patch_latent_dim": generator_config.patch_latent_dim,
        },
    )
    builder.set_metadata(
        "conditioning_handoffs",
        {
            "video": {
                "from": f"{_VIDEO_ENCODER_NAME}.latent",
                "to": f"{_GENERATOR_NAME}.vision_tokens",
                "transform": "patchify",
                "parameters": {
                    "spatial_patch_size": generator_config.latent_patch_size,
                    "temporal_patch_size": 1,
                    "input_layout": "BCTHW",
                    "output_layout": "NC",
                    "channel_order": "patch_width_patch_height_channel",
                },
                "optional": True,
            },
            "audio": (
                {
                    "from": f"{_AUDIO_ENCODER_NAME}.latent_mean",
                    "to": f"{_GENERATOR_NAME}.sound_tokens",
                    "transform": "reshape",
                    "parameters": {
                        "input_layout": "BCT",
                        "output_layout": "TC",
                    },
                    "optional": True,
                }
                if audio_package is not None and "encoder" in audio_package
                else None
            ),
        },
    )
    builder.set_metadata(
        "shared_parameters",
        {
            "understanding_expert": [
                _REASONER_NAMES["decoder"],
                _GENERATOR_NAME,
            ]
        },
    )
    builder.set_metadata(
        "prompt_inputs",
        {
            "reasoner": f"{_REASONER_NAMES['embedding']}.input_ids",
            "generator": f"{_GENERATOR_NAME}.input_ids",
            "relationship": "same_tokenized_prompt_repacked_for_generator",
        },
    )
    if generator_config.action_gen:
        builder.set_metadata(
            "action",
            {
                "modes": ["policy", "forward_dynamics", "inverse_dynamics"],
                "domain_ids": _ACTION_DOMAIN_IDS,
                "raw_dimensions": _ACTION_RAW_DIMS,
                "padded_dimension": generator_config.action_dim,
                "input_padding": {
                    "side": "right",
                    "value": 0.0,
                    "target_dimension": generator_config.action_dim,
                },
                "output_slicing": "raw_dimensions",
                "clipping": None,
                "resolution_tiers": [256, 480, 704, 720],
            },
        )
    for key, value in (extra_metadata or {}).items():
        builder.set_metadata(key, value)

    return builder.build(
        config=reasoner_package.config,
        component_configs=component_configs,
    )


def _collect_assets(
    model_id: str,
    *,
    has_sound_tokenizer: bool,
) -> dict[str, tuple[str, bool]]:
    assets: dict[str, tuple[str, bool]] = {}
    candidates = list(_ASSET_CANDIDATES)
    if has_sound_tokenizer:
        candidates.append(("sound_tokenizer/config.json", True))
    for destination, required in candidates:
        source = _resolve_file(model_id, destination, required=required)
        if source is not None:
            assets[destination] = (source, required)
    tokenizer_paths = ("tokenizer.json", "text_tokenizer/tokenizer.json")
    available_tokenizers = [path for path in tokenizer_paths if path in assets]
    if not available_tokenizers:
        raise FileNotFoundError(
            "Cosmos3 checkpoint must provide tokenizer.json either at the root "
            "or under text_tokenizer/."
        )
    preferred = available_tokenizers[0]
    assets[preferred] = (assets[preferred][0], True)
    return assets


def build_cosmos3_world_model(
    model_id: str,
    *,
    dtype: str | ir.DataType | None = None,
    load_weights: bool = True,
    execution_provider: str = "default",
    trace_optimization: bool = False,
    **_options: Any,
) -> PipelinePackage:
    """Build the complete neural Cosmos3-Omni world-model package."""
    pipeline_index, _ = _load_json(model_id, "model_index.json")
    root_config, _ = _load_json(model_id, "config.json")
    text_config = root_config.get("text_config") or {}
    if (
        isinstance(text_config, Mapping)
        and text_config.get("model_type") == "cosmos3_edge_text"
    ):
        from mobius._cosmos3_edge_world_model import build_cosmos3_edge_world_model

        return build_cosmos3_edge_world_model(
            model_id,
            dtype=dtype,
            load_weights=load_weights,
            execution_provider=execution_provider,
            trace_optimization=trace_optimization,
            **_options,
        )
    transformer_class = _component_class(pipeline_index, "transformer")
    vae_class = _component_class(pipeline_index, "vae")
    if transformer_class != "Cosmos3OmniTransformer":
        raise ValueError(f"Unsupported Cosmos3 transformer class {transformer_class!r}")
    if vae_class != "AutoencoderKLWan":
        raise ValueError(f"Unsupported Cosmos3 VAE class {vae_class!r}")

    transformer_config_dict, _ = _load_json(model_id, "transformer/config.json")
    vae_config_dict, _ = _load_json(model_id, "vae/config.json")
    scheduler_config, _ = _load_json(model_id, "scheduler/scheduler_config.json")
    generation_config = _load_optional_json(model_id, "generation_config.json")
    checkpoint_config = _load_optional_json(model_id, "checkpoint.json")
    policy_config = checkpoint_config.get("policy")
    default_action_domain = (
        policy_config.get("domain_name", "no_action")
        if isinstance(policy_config, Mapping)
        else "no_action"
    )

    has_sound = _component_class(pipeline_index, "sound_tokenizer") is not None
    has_reasoner_vision = _component_class(pipeline_index, "vision_encoder") is not None
    audio_config_dict: dict[str, Any] | None = None
    audio_weight_names: set[str] | None = None
    if has_sound:
        audio_config_dict, _ = _load_json(model_id, "sound_tokenizer/config.json")
        audio_weight_names = _component_weight_names(model_id, "sound_tokenizer")

    (
        reasoner_package,
        reasoner_module,
        generator_package,
        generator_module,
        vae_package,
        vae_module,
        audio_package,
        audio_module,
    ) = _build_components(
        model_id,
        dtype=dtype,
        execution_provider=execution_provider,
        trace_optimization=trace_optimization,
        pipeline_index=pipeline_index,
        transformer_config_dict=transformer_config_dict,
        vae_config_dict=vae_config_dict,
        audio_config_dict=audio_config_dict,
        audio_weight_names=audio_weight_names,
        has_reasoner_vision=has_reasoner_vision,
    )

    if load_weights:
        _apply_checkpoint_weights(
            model_id,
            reasoner_package=reasoner_package,
            reasoner_module=reasoner_module,
            generator_package=generator_package,
            generator_module=generator_module,
            vae_package=vae_package,
            vae_module=vae_module,
            audio_package=audio_package,
            audio_module=audio_module,
        )

    assets = _collect_assets(model_id, has_sound_tokenizer=has_sound)
    return _compose_pipeline(
        model_id=model_id,
        reasoner_package=reasoner_package,
        generator_package=generator_package,
        vae_package=vae_package,
        audio_package=audio_package,
        generator_config=generator_module.config,
        vae_config=vae_module.config,
        scheduler_config=scheduler_config,
        assets=assets,
        execution_provider=execution_provider,
        generation_config=generation_config,
        default_inference_steps=(4 if "4Step" in model_id else 35),
        default_action_domain=default_action_domain,
    )
