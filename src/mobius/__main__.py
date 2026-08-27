# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Command-line interface for mobius."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import tqdm

if TYPE_CHECKING:
    import onnx_ir as ir
    import torch

from mobius._builder import (
    DTYPE_MAP,
    build_from_module,
    resolve_dtype,
)
from mobius._optimizations import strip_debug_metadata
from mobius._registry import registry
from mobius.integrations.transformers import (
    _config_from_hf,
    _default_task_for_model,
    build,
)

logger = logging.getLogger(__name__)


# Rust/cargo-style build features. Each feature name (kebab-case) maps to the
# boolean attribute on the parsed args that it enables. `--features a,b` (or a
# repeated `--features`) is the canonical way to toggle these build modes.
_BUILD_FEATURES: dict[str, str] = {
    "static-cache": "static_cache",
    "fp8-kv-cache": "fp8_kv_cache",
    "prune-prefill-prefix": "prune_prefill_prefix",
    "text-only": "text_only",
    "glm-full-attention": "glm_full_attention",
    "paged-attention": "export_paged_attention",
}


def _resolve_build_features(args: argparse.Namespace) -> None:
    """Fold ``--features`` values into the boolean build-mode attributes.

    ``--features`` accepts a comma-separated list and may be repeated (cargo
    style), e.g. ``--features fp8-kv-cache,static-cache`` or
    ``--features fp8-kv-cache --features static-cache``. Each recognised feature
    sets its corresponding attribute (``fp8-kv-cache`` -> ``args.fp8_kv_cache``).
    Unknown feature names raise ``SystemExit``.
    """
    for dest in _BUILD_FEATURES.values():
        if not hasattr(args, dest):
            setattr(args, dest, False)

    raw = getattr(args, "features", None) or []
    requested: list[str] = []
    for chunk in raw:
        requested.extend(f.strip() for f in chunk.split(",") if f.strip())

    for feature in requested:
        dest = _BUILD_FEATURES.get(feature)
        if dest is None:
            valid = ", ".join(sorted(_BUILD_FEATURES))
            raise SystemExit(f"Error: unknown feature '{feature}'. Valid features: {valid}.")
        setattr(args, dest, True)


def _parse_size(size_str: str) -> int:
    """Parse a human-readable size string (e.g. '5GB') to bytes."""
    size_str = size_str.strip().upper()
    multipliers = {"B": 1, "KB": 1000, "MB": 1000**2, "GB": 1000**3, "TB": 1000**4}
    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if size_str.endswith(suffix):
            return int(float(size_str[: -len(suffix)]) * mult)
    return int(size_str)


def _load_weights_from_dir(model_dir: str) -> dict[str, torch.Tensor]:
    """Load safetensors weights from a local model directory."""
    import safetensors.torch

    model_dir = os.path.abspath(model_dir)
    if os.path.isfile(model_dir):
        model_dir = os.path.dirname(model_dir)

    index_path = os.path.join(model_dir, "model.safetensors.index.json")

    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        all_files = sorted(set(index["weight_map"].values()))
        paths = [os.path.join(model_dir, f) for f in all_files]
    else:
        paths = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))

    if not paths:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")

    state_dict: dict[str, torch.Tensor] = {}
    for path in tqdm.tqdm(paths, desc="Loading weights"):
        state_dict.update(safetensors.torch.load_file(path))
    return state_dict


def _apply_optimize(model: ir.Model, optimize: str | None) -> None:
    """Apply rewrite rules if --optimize is specified."""
    if not optimize:
        return

    from onnxscript.rewriter import rewrite

    from mobius.rewrite_rules import (
        bias_gelu_rules,
        group_query_attention_rules,
        packed_attention_rules,
        skip_layer_norm_rules,
        skip_norm_rules,
    )

    rule_map = {
        "bias_gelu": bias_gelu_rules,
        "group_query_attention": group_query_attention_rules,
        "packed_attention": packed_attention_rules,
        "skip_layer_norm": skip_layer_norm_rules,
        "skip_norm": skip_norm_rules,
    }

    if optimize == "all":
        rule_names = list(rule_map)
    else:
        rule_names = [r.strip() for r in optimize.split(",")]
        for name in rule_names:
            if name not in rule_map:
                raise ValueError(
                    f"Unknown rewrite rule '{name}'. Available: {sorted(rule_map)}"
                )

    for name in rule_names:
        rules = rule_map[name]()
        rewrite(model, pattern_rewrite_rules=rules)
        print(f"Applied rewrite rule: {name}")


def _cmd_build(args: argparse.Namespace) -> None:
    """Execute the 'build' subcommand."""
    import dataclasses

    from mobius.integrations.diffusers._builder import (
        _load_diffusers_pipeline_index,
        build_diffusers_pipeline,
    )
    from mobius.tasks import CausalLMTask, ModelTask

    def _resolve_static_cache_task(model_type: str) -> ModelTask:
        """Create the correct static cache task for the given model type.

        ``--features text-only`` makes :func:`build` swap the checkpoint's
        multimodal ``model_type`` for its text-only registry sibling, so the
        task must be resolved against the *same* substituted type. Resolving
        against the raw checkpoint type instead pairs a text-only module with a
        multimodal task, which then fails looking for sub-modules (a vision
        tower, a separate decoder) that a text-only module does not have.
        """
        if args.text_only:
            from mobius._registry import _TEXT_ONLY_MODEL_TYPE

            model_type = _TEXT_ONLY_MODEL_TYPE.get(model_type, model_type)
        if model_type == "falcon_h1":
            raise ValueError(
                "--static-cache cannot represent Falcon-H1's per-layer K, V, "
                "convolution, and SSM states"
            )
        if model_type == "gemma4":
            from mobius.tasks._gemma4 import Gemma4Task

            return Gemma4Task(
                static_cache=True,
                max_seq_len=args.max_seq_len,
            )
        if model_type == "gemma4_text":
            from mobius.tasks._gemma4 import Gemma4TextCausalLMTask

            return Gemma4TextCausalLMTask(
                static_cache=True,
                max_seq_len=args.max_seq_len,
            )
        return CausalLMTask(static_cache=True, max_seq_len=args.max_seq_len)

    # Fold --features into the boolean build-mode attributes before any
    # validation reads them.
    _resolve_build_features(args)

    # Validate --max-seq-len requires the static-cache feature.
    if args.max_seq_len is not None and not args.static_cache:
        raise SystemExit("Error: --max-seq-len can only be used with --features static-cache.")

    # Validate --max-seq-len is positive
    if args.max_seq_len is not None and args.max_seq_len <= 0:
        raise SystemExit("Error: --max-seq-len must be a positive integer.")
    if args.max_workers <= 0:
        raise SystemExit("Error: --max-workers must be a positive integer.")

    max_length = getattr(args, "max_length", None)
    if max_length is not None and args.runtime != "onnx-genai":
        raise SystemExit("Error: --max-length can only be used with --runtime onnx-genai.")
    if max_length is not None and max_length <= 0:
        raise SystemExit("Error: --max-length must be a positive integer.")
    guidance_scale = getattr(args, "guidance_scale", None)
    if guidance_scale is not None and args.runtime != "onnx-genai":
        raise SystemExit("Error: --guidance-scale can only be used with --runtime onnx-genai.")

    # Validate static-cache + --task compatibility.
    if args.static_cache and args.task is not None:
        raise SystemExit(
            "Error: --features static-cache cannot be combined with --task. "
            "Remove --task to use --features static-cache."
        )

    # text-only resolution lives in build() (model_type remap + config
    # stripping), which is only reached on the HuggingFace model-ID path.
    if args.text_only and args.config:
        raise SystemExit(
            "Error: --features text-only is not supported with --config (local "
            "directory). Use --model <hf-id> --features text-only instead."
        )
    if args.revision is not None and args.config:
        raise SystemExit(
            "Error: --revision is only supported with --model. "
            "Local --config directories are already immutable inputs."
        )

    # --component selects one component of a diffusers pipeline; text-only
    # produces a single decoder-only model. Combining them would silently
    # filter that model away unless --component happens to be 'model'.
    if args.text_only and args.component:
        raise SystemExit(
            "Error: --features text-only is not supported with --component. "
            "The text-only feature produces a single decoder-only model, while "
            "--component selects a component of a diffusers pipeline."
        )

    load_weights = not args.no_weights
    keep_quantized = not getattr(args, "dequantize", False)
    task: str | ModelTask | None = args.task

    # FP8 KV cache: resolve the optional per-layer scale file up front so both
    # the --config and --model build paths can pass the same scales.
    fp8_kv_cache = getattr(args, "fp8_kv_cache", False)
    prune_prefill_prefix = getattr(args, "prune_prefill_prefix", False)
    kv_cache_scales: dict[int, tuple[float, float]] | None = None
    scale_file = getattr(args, "kv_cache_scale_file", None)
    if scale_file is not None and not fp8_kv_cache:
        raise SystemExit(
            "Error: --kv-cache-scale-file can only be used with --features fp8-kv-cache."
        )
    if fp8_kv_cache and scale_file is not None:
        from mobius._passes._fp8_kv_cache import load_kv_cache_scale_file

        kv_cache_scales = load_kv_cache_scale_file(scale_file)

    if args.static_cache:
        # Defer task creation — we need to know the model type first.
        # Store parameters for later resolution.
        static_cache_params = {
            "static_cache": True,
            "max_seq_len": args.max_seq_len,
        }
    else:
        static_cache_params = None

    # PagedAttention (LATENT dense-MLA) export uses the paged-cache task with
    # caller-owned page buffers. It is a distinct cache authority, so it cannot
    # be combined with the static-cache task or an explicit --task.
    export_paged_attention = getattr(args, "export_paged_attention", False)
    if export_paged_attention:
        if static_cache_params is not None:
            raise SystemExit(
                "Error: --features paged-attention cannot be combined with "
                "--features static-cache."
            )
        if task is not None:
            raise SystemExit(
                "Error: --features paged-attention cannot be combined with --task. "
                "Remove --task to use --features paged-attention."
            )
        task = CausalLMTask(paged_cache=True)
    trust_remote_code = args.trust_remote_code
    revision = args.revision
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    dtype_override = resolve_dtype(args.dtype)
    optimize = args.optimize
    component_filter = args.component
    execution_provider = args.execution_provider

    # Auto-detect diffusers pipelines. Skipped when the text-only feature is set:
    # that flag only applies to transformers decoder exports, so we let the
    # central build() validation reject a diffusers/unsupported repo rather
    # than silently exporting a diffusion pipeline and ignoring the flag.
    if args.model and not args.config and not args.text_only:
        pipeline_index = _load_diffusers_pipeline_index(args.model, revision=revision)
        if pipeline_index is not None:
            print(
                f"Detected diffusers pipeline: {pipeline_index.get('_class_name', 'Unknown')}"
            )
            pipeline_components = None
            if component_filter:
                roots = [
                    name
                    for name in pipeline_index
                    if not name.startswith("_")
                    and (component_filter == name or component_filter.startswith(f"{name}_"))
                ]
                pipeline_components = {max(roots, key=len)} if roots else {component_filter}
            pkg = build_diffusers_pipeline(
                args.model,
                revision=revision,
                dtype=dtype_override,
                load_weights=load_weights,
                components=pipeline_components,
                execution_provider=execution_provider,
            )
            _save_package(pkg, output_dir, args, optimize, component_filter)
            return

    # Auto-detect NeMo .nemo archives (local file or HF ref like
    # 'owner/repo:model.nemo'). Routes to the NeMo import path; reuses the
    # standard build args (--dtype, --ep, --external-data) and save logic.
    if args.model and args.model.endswith(".nemo"):
        from mobius.integrations.nemo import build_from_nemo

        print(f"Detected NeMo archive: {args.model}")
        pkg = build_from_nemo(
            args.model,
            revision=args.revision,
            dtype=dtype_override,
            execution_provider=execution_provider,
        )
        _save_package(pkg, output_dir, args, optimize, component_filter)
        return

    # Build from HuggingFace model ID or local config
    if args.config:
        import onnx_ir as ir
        import transformers

        config_path = args.config
        try:
            hf_config = transformers.AutoConfig.from_pretrained(
                config_path, trust_remote_code=trust_remote_code
            )
        except (ValueError, KeyError, OSError):
            # A checkpoint predating the mandatory ``model_type`` key still
            # names its architecture; resolve it the same way the HF-id path
            # does rather than refusing a directory Mobius can build.
            from mobius.integrations.transformers._config_resolver import (
                _try_load_config_json,
            )

            hf_config = _try_load_config_json(config_path)
            if hf_config is None:
                raise
        model_type = hf_config.model_type
        parent_config = hf_config
        from mobius.integrations.transformers._builder import (
            _is_qwen4_exp_composite,
        )

        if _is_qwen4_exp_composite(parent_config):
            raise SystemExit(
                "Error: local Qwen4-Exp composite configs cannot be silently "
                "exported as text-only. Use --model <hf-id> --features text-only."
            )
        if hasattr(hf_config, "text_config"):
            hf_config = hf_config.text_config
        from mobius.integrations.compressed_tensors import (
            CompressedTensorsConfig,
            stream_compressed_tensors_to_package,
        )

        compressed_tensors_config = CompressedTensorsConfig.from_hf_config(parent_config)
        module_class = registry.get(model_type)
        config = _config_from_hf(
            hf_config,
            parent_config=parent_config,
            module_class=module_class,
        )
        if dtype_override is not None:
            config = dataclasses.replace(config, dtype=dtype_override)
        elif compressed_tensors_config is not None and keep_quantized:
            config = dataclasses.replace(config, dtype=ir.DataType.FLOAT16)
        if (
            compressed_tensors_config is not None
            and keep_quantized
            and config.dtype != ir.DataType.FLOAT16
        ):
            raise SystemExit(
                "Error: storage-preserving compressed-tensors export requires "
                "--dtype f16 for the Microsoft W4A16/W8A16 custom-op ABI. "
                "Use --dtype f16 or --dequantize."
            )
        if args.glm_full_attention:
            if model_type != "glm_moe_dsa":
                raise SystemExit(
                    "Error: --features glm-full-attention is only supported for "
                    f"model_type 'glm_moe_dsa' (got '{model_type}')."
                )
            config = dataclasses.replace(config, use_dsa=False)
        if export_paged_attention:
            from mobius.components._paged_mla import paged_attention_rejection

            config = dataclasses.replace(config, export_paged_attention=True)
            reason = paged_attention_rejection(config)
            if reason is not None:
                raise SystemExit(
                    f"Error: --features paged-attention is not supported for this "
                    f"model: {reason}"
                )
        if static_cache_params is not None:
            task = _resolve_static_cache_task(model_type)
        elif task is None:
            task = _default_task_for_model(model_type)
        from mobius.tasks import get_task

        resolved_task = get_task(task)
        component_manifest = resolved_task.component_manifest(
            module_class=module_class,
            model_type=model_type,
            hf_config=parent_config,
        )
        model_module = module_class(config)
        pkg = build_from_module(
            model_module,
            config,
            task=task,
            execution_provider=execution_provider,
            fp8_kv_cache=fp8_kv_cache,
            kv_cache_scales=kv_cache_scales,
            prune_prefill_prefix=prune_prefill_prefix,
            component_manifest=component_manifest,
        )
        for name, model in pkg.items():
            model.graph.name = f"{config_path}/{name}"
        if load_weights:
            if compressed_tensors_config is not None:
                # Packed FP4 weights cannot pass through ordinary apply_weights.
                # The same loader owns both faithful native storage and the
                # explicit keep_quantized=False dense reconstruction policy.
                checkpoint_dir = (
                    os.path.dirname(config_path)
                    if os.path.isfile(config_path)
                    else config_path
                )
                stream_compressed_tensors_to_package(
                    pkg,
                    checkpoint_dir,
                    compressed_tensors_config,
                    preprocess_weights=getattr(
                        model_module,
                        "preprocess_weights",
                        None,
                    ),
                    fp8_kv_cache=fp8_kv_cache,
                    keep_quantized=keep_quantized,
                )
            else:
                state_dict = _load_weights_from_dir(config_path)
                if hasattr(model_module, "preprocess_weights"):
                    state_dict = model_module.preprocess_weights(state_dict)
                from mobius._component_quantization import (
                    normalize_component_quantized_weights,
                )

                state_dict = normalize_component_quantized_weights(
                    state_dict,
                    model_module,
                    config,
                    pkg.keys(),
                    manifest=component_manifest,
                    task=resolved_task,
                )
                pkg.apply_weights(state_dict)
    else:
        model_id_or_path = args.model
        if static_cache_params is not None:
            # Detect model type to resolve the correct static cache task.
            import transformers

            hf_config = transformers.AutoConfig.from_pretrained(
                model_id_or_path,
                trust_remote_code=trust_remote_code,
                revision=revision,
            )
            task = _resolve_static_cache_task(getattr(hf_config, "model_type", ""))

        pkg = build(
            model_id_or_path,
            task=task,
            dtype=dtype_override,
            load_weights=load_weights,
            revision=revision,
            trust_remote_code=trust_remote_code,
            execution_provider=execution_provider,
            text_only=args.text_only,
            fp8_kv_cache=fp8_kv_cache,
            kv_cache_scales=kv_cache_scales,
            prune_prefill_prefix=prune_prefill_prefix,
            glm_full_attention=args.glm_full_attention,
            export_paged_attention=export_paged_attention,
            keep_quantized=keep_quantized,
        )

    _save_package(pkg, output_dir, args, optimize, component_filter)


def _save_package(
    pkg, output_dir: str, args, optimize: str | None, component_filter: str | None
) -> None:
    """Save a ModelPackage to disk, applying optimizations and runtime configs."""
    runtime = getattr(args, "runtime", None)
    if runtime == "ort-genai":
        from mobius.integrations.ort_genai.auto_export import (
            _validate_ort_genai_compatibility,
        )

        try:
            _validate_ort_genai_compatibility(pkg)
        except ValueError as error:
            raise SystemExit(f"Error: {error}") from error

    components = (lambda name: name == component_filter) if component_filter else None
    for name, model in pkg.items():
        if components is not None and not components(name):
            continue
        _apply_optimize(model, optimize)
        if args.release:
            # Last thing before saving, so metadata that later stages read (and
            # that rewrite rules add as they run) is still present while they
            # need it.
            strip_debug_metadata(model)

    max_shard_size_bytes = _parse_size(args.max_shard_size) if args.max_shard_size else None

    if args.external_data == "safetensors" and args.execution_provider == "cuda":
        logging.getLogger(__name__).warning(
            "Safetensors external data does not guarantee 256-byte offset "
            "alignment, which can cause CUBLAS misaligned address errors on "
            "CUDA. Consider using --external-data onnx for CUDA builds."
        )

    pkg.save(
        output_dir,
        external_data=args.external_data,
        max_shard_size_bytes=max_shard_size_bytes,
        max_workers=args.max_workers,
        components=components,
        check_weights=not args.no_weights,
    )
    selected = [name for name in pkg if components is None or components(name)]
    use_subfolders = len(selected) > 1
    for name in selected:
        if use_subfolders:
            path = os.path.join(output_dir, name, "model.onnx")
        else:
            path = os.path.join(output_dir, "model.onnx")
        print(f"Saved {name} to {path}")

    if runtime == "ort-genai":
        from mobius.integrations.ort_genai import write_ort_genai_config

        hf_model_id = getattr(args, "model", None)
        ep = getattr(args, "execution_provider", "cpu")
        # When --config (local dir) is used instead of --model, copy tokenizer
        # files from the local directory rather than downloading from HF.
        local_config_dir = getattr(args, "config", None)
        artifacts = write_ort_genai_config(
            pkg,
            output_dir,
            hf_model_id=hf_model_id,
            ep=ep,
            local_config_dir=local_config_dir,
            trust_remote_code=getattr(args, "trust_remote_code", False),
            revision=getattr(args, "revision", None),
        )
        for name, path in artifacts.items():
            print(f"  {name}: {path}")
    elif runtime == "onnx-genai":
        from mobius.integrations.onnx_genai import write_onnx_genai_config
        from mobius.integrations.onnx_genai.inference_metadata import (
            is_native_vlm_package,
            write_native_vlm_package_metadata,
        )

        config = getattr(pkg, "config", None)
        source = getattr(args, "config", None) or getattr(args, "model", None)
        revision = getattr(args, "revision", None)
        if is_native_vlm_package(pkg):
            try:
                artifacts = write_native_vlm_package_metadata(
                    pkg,
                    output_dir,
                    config=config,
                    source=source,
                    revision=revision,
                )
            except ValueError as error:
                raise SystemExit(f"Error: {error}") from error
        else:
            try:
                artifacts = write_onnx_genai_config(
                    pkg,
                    output_dir,
                    config=config,
                    source=source,
                    revision=revision,
                    guidance_scale=getattr(args, "guidance_scale", None),
                )
            except ValueError as error:
                raise SystemExit(f"Error: {error}") from error
        for name, path in artifacts.items():
            print(f"  {name}: {path}")


def _cmd_list(args: argparse.Namespace) -> None:
    """Execute the 'list' subcommand."""
    from mobius.tasks import TASK_REGISTRY

    resource = args.resource
    if resource == "models":
        architectures = registry.architectures()
        print(f"Supported model architectures ({len(architectures)}):\n")
        for arch in architectures:
            module_class = registry.get(arch)
            task = getattr(module_class, "default_task", "text-generation")
            category = getattr(module_class, "category", "")
            print(f"  {arch:<30} task={task:<25} category={category}")
    elif resource == "tasks":
        print(f"Available tasks ({len(TASK_REGISTRY)}):\n")
        for name in sorted(TASK_REGISTRY):
            cls = TASK_REGISTRY[name]
            print(f"  {name:<35} {cls.__name__}")
    elif resource == "dtypes":
        seen: set[str] = set()
        print("Available dtypes:\n")
        for _name, dt in sorted(DTYPE_MAP.items()):
            if dt.name not in seen:
                aliases = [k for k, v in DTYPE_MAP.items() if v == dt]
                print(f"  {' | '.join(aliases):<25} → {dt.name}")
                seen.add(dt.name)
    elif resource == "eps":
        from mobius._execution_providers import ep_registry

        eps = sorted(ep_registry.names())
        print(f"Registered execution providers ({len(eps)}):\n")
        for ep_name in eps:
            caps = ep_registry.require(ep_name)
            gqa = ", ".join(dt.name for dt in sorted(caps.gqa_dtypes, key=lambda d: d.name))
            extras = []
            if not caps.supports_fused_rope:
                extras.append("no-fused-rope")
            if not caps.supports_skip_layer_norm:
                extras.append("no-skip-layer-norm")
            flags = f"  [{', '.join(extras)}]" if extras else ""
            print(f"  {ep_name:<12} gqa_dtypes=[{gqa}]{flags}")
    else:
        print(f"Unknown resource '{resource}'. Use: models, tasks, dtypes, eps")


def _cmd_build_gguf(args: argparse.Namespace) -> None:
    """Execute the 'build-gguf' subcommand."""
    try:
        from mobius.integrations.gguf import build_from_gguf
    except ImportError:
        print(
            "GGUF support requires the gguf package. Install with: pip install mobius-onnx[gguf]"
        )
        raise SystemExit(1)

    mmproj_path = getattr(args, "mmproj", None)
    keep_quantized = not args.dequantize
    reuse_gguf_weights = args.reuse_gguf_weights
    if reuse_gguf_weights and args.runtime == "ort-genai":
        raise SystemExit(
            "Error: --reuse-gguf-weights cannot be combined with --runtime ort-genai "
            "because genai_config.json cannot require disabled ORT constant folding. "
            "Use direct ONNX Runtime with ORT_DISABLE_ALL."
        )

    if keep_quantized:
        print(
            "Quantized-target mode: classifying GGUF qtypes before conversion; "
            "lossy requantization will be reported..."
        )
    else:
        print(
            "Dequantized mode: converting GGUF weights to explicitly reported float storage..."
        )

    gguf_path = args.gguf_path
    gguf_reference = gguf_path
    output_dir = args.output_dir
    target_config = getattr(args, "target_config", None)
    runtime = getattr(args, "runtime", None)

    if args.max_seq_len is not None and not args.static_cache:
        raise SystemExit("Error: --max-seq-len can only be used with --static-cache.")
    if args.max_seq_len is not None and args.max_seq_len <= 0:
        raise SystemExit("Error: --max-seq-len must be a positive integer.")
    if args.max_workers <= 0:
        raise SystemExit("Error: --max-workers must be a positive integer.")
    if mmproj_path is not None and args.static_cache:
        raise SystemExit("Error: --static-cache cannot be used with --mmproj.")
    if target_config is not None and runtime is not None:
        raise SystemExit(
            "Error: dflash/eagle3 target-coupled drafts do not support standalone "
            "runtime packaging; omit --runtime to save the auxiliary graph and manifest."
        )
    tokenizer_repository = getattr(args, "tokenizer_repository", None)
    tokenizer_revision = getattr(args, "tokenizer_revision", None)
    if (tokenizer_repository is None) != (tokenizer_revision is None):
        raise SystemExit(
            "Error: --tokenizer-repository and --tokenizer-revision must be provided together."
        )
    if runtime is None and tokenizer_repository is not None:
        raise SystemExit(
            "Error: pinned tokenizer materialization is only available with --runtime."
        )

    if runtime is not None:
        from mobius.integrations.gguf._arch_registry import get_arch_spec
        from mobius.integrations.gguf._builder import (
            _resolve_gguf_path,
            _validate_gguf_model,
        )
        from mobius.integrations.gguf._shard_set import open_gguf_model
        from mobius.integrations.gguf._spec import Support

        # Resolve and validate the exact selected source before graph construction
        # so a deferred tokenizer cannot leave a graph-only directory behind.
        resolved_gguf_path = _resolve_gguf_path(gguf_path)
        gguf_model = open_gguf_model(resolved_gguf_path)
        _validate_gguf_model(gguf_model, source=str(resolved_gguf_path))
        architecture_spec = get_arch_spec(gguf_model.architecture)
        if architecture_spec.runtime is not Support.SUPPORTED:
            raise SystemExit(
                f"Error: GGUF runtime packaging for {architecture_spec.gguf_arch!r} is "
                f"{architecture_spec.runtime.value}: {architecture_spec.reason}"
            )
        if tokenizer_repository is None or tokenizer_revision is None:
            raise SystemExit(
                "Error: GGUF runtime packaging requires --tokenizer-repository and an "
                "immutable --tokenizer-revision."
            )
        if tokenizer_repository.count("/") != 1 or not all(tokenizer_repository.split("/")):
            raise SystemExit(
                "Error: --tokenizer-repository must be an owner/repository Hub ID."
            )
        if re.fullmatch(r"[0-9a-f]{40}", tokenizer_revision) is None:
            raise SystemExit(
                "Error: --tokenizer-revision must be an immutable 40-hex commit SHA."
            )

    pkg = build_from_gguf(
        gguf_reference,
        mmproj=mmproj_path,
        image_token_id=args.image_token_id,
        dtype=args.dtype,
        keep_quantized=keep_quantized,
        execution_provider=args.execution_provider,
        static_cache=args.static_cache,
        max_seq_len=args.max_seq_len,
        reuse_gguf_weights=reuse_gguf_weights,
        target_config=target_config,
        _gguf_model=gguf_model if runtime is not None else None,
    )

    if args.release:
        for model in pkg.values():
            strip_debug_metadata(model)

    if runtime is None:
        os.makedirs(output_dir, exist_ok=True)
        pkg.save(
            output_dir,
            external_data=args.external_data,
            max_shard_size_bytes=(
                _parse_size(args.max_shard_size) if args.max_shard_size else None
            ),
            max_workers=args.max_workers,
        )
        _print_saved_gguf_models(pkg, output_dir)

        # ModelPackage.save() persisted the MTP sidecar into its manifest-selected
        # collision-safe directory.
        mtp_head = getattr(pkg, "mtp_head", None)
        if mtp_head is not None:
            from mobius._model_package import _read_mtp_sidecar_name

            mtp_dir = _read_mtp_sidecar_name(output_dir)
            assert mtp_dir is not None
            print(f"Saved mtp head to {os.path.join(output_dir, mtp_dir, 'model.onnx')}")

        draft_manifest = getattr(pkg, "draft_manifest", None)
        if draft_manifest is not None:
            from mobius.integrations.gguf._draft import write_draft_manifest

            manifest_path = write_draft_manifest(draft_manifest, output_dir)
            print(f"Saved draft pairing manifest to {manifest_path}")

    if runtime in ("onnx-genai", "ort-genai"):
        from mobius.integrations.gguf import write_gguf_runtime_package

        artifacts = write_gguf_runtime_package(
            pkg,
            gguf_path,
            output_dir,
            runtime=runtime,
            runtime_version=getattr(args, "runtime_version", None),
            tokenizer_repository=tokenizer_repository,
            tokenizer_revision=tokenizer_revision,
            local_files_only=getattr(args, "local_files_only", False),
            external_data=args.external_data,
            max_shard_size_bytes=(
                _parse_size(args.max_shard_size) if args.max_shard_size else None
            ),
            max_workers=args.max_workers,
        )
        # The writer returns only after atomically publishing the complete graph,
        # tokenizer, and runtime configuration directory.
        _print_saved_gguf_models(pkg, output_dir)
        for name, path in artifacts.items():
            print(f"  {name}: {path}")


def _print_saved_gguf_models(pkg: Any, output_dir: str) -> None:
    """Report model paths only after their containing package is durable."""
    use_subfolders = len(pkg) > 1
    for name in pkg:
        path = (
            os.path.join(output_dir, name, "model.onnx")
            if use_subfolders
            else os.path.join(output_dir, "model.onnx")
        )
        print(f"Saved {name} to {path}")


def _cmd_convert_comfyui(args: argparse.Namespace) -> None:
    """Execute the 'convert-comfyui' subcommand."""
    from mobius.integrations.onnx_genai import convert_comfyui_workflow

    with open(args.workflow, encoding="utf-8") as handle:
        workflow = json.load(handle)
    result = convert_comfyui_workflow(
        workflow,
        args.checkpoint,
        args.output,
        sdxl=getattr(args, "sdxl", False),
        revision=args.revision,
    )
    wf = result.workflow
    print(f"Converted ComfyUI workflow -> {result.output_dir}")
    print(f"  metadata: {result.metadata_path}")
    print(f"  run params: {result.run_params_path}")
    print(
        f"  {wf.steps} steps, cfg {wf.cfg}, sampler {wf.sampler_name} "
        f"(scheduler {wf.scheduler_kind}), {wf.width}x{wf.height}"
    )
    if wf.loras:
        print(f"  loras: {', '.join(f'{n}@{s}' for n, s in wf.loras)}")
    if wf.prompt is not None:
        print(f"  prompt: {wf.prompt!r}")


def _cmd_preflight_gguf(args: argparse.Namespace) -> None:
    """Execute the 'preflight-gguf' subcommand (metadata only)."""
    try:
        from mobius.integrations.gguf import preflight_gguf
    except ImportError:
        print(
            "GGUF support requires the gguf package. Install with: pip install mobius-onnx[gguf]"
        )
        raise SystemExit(1)

    report = preflight_gguf(
        args.source,
        filename=getattr(args, "filename", None),
        revision=getattr(args, "revision", None),
        verify_checksums=getattr(args, "verify_checksums", False),
        cache_path=getattr(args, "cache", None),
    )
    if getattr(args, "json", False):
        print(report.to_json())
    else:
        print(report.render())
    if report.blockers:
        raise SystemExit(2)


def _probe_gpu_total_bytes() -> int | None:
    """Return the largest single-GPU memory in bytes via nvidia-smi, or None."""
    import shutil as _shutil
    import subprocess

    exe = _shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    mibs = [int(line) for line in out.stdout.split() if line.strip().isdigit()]
    if not mibs:
        return None
    return max(mibs) * 1024 * 1024


def _cmd_preflight(args: argparse.Namespace) -> None:
    """Execute the 'preflight' subcommand (resumable export dry-run)."""
    from mobius.preflight import ExportMode, LoaderMode, run_preflight

    gpu_total = None
    if args.gpu_total_bytes:
        gpu_total = _parse_size(args.gpu_total_bytes)
    elif not args.no_gpu_probe:
        gpu_total = _probe_gpu_total_bytes()

    result = run_preflight(
        args.model_id,
        output_dir=args.output,
        revision=args.revision,
        download_dir=args.download_dir,
        export_mode=ExportMode(args.export_mode),
        loader=LoaderMode(args.loader),
        group_size=args.group_size,
        target_dtype_bytes=args.target_dtype_bytes,
        gpu_total_bytes=gpu_total,
        margin_frac=args.margin,
        state_path=args.state,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        b = result.budget
        print(f"Model:        {result.model_id}")
        print(f"Revision:     {result.revision or 'default'}")
        print(f"Commit:       {result.commit_sha or '(local)'}")
        print(f"Shards:       {len(result.shards)}")
        if b is not None:
            print(f"Params:       {b.param_count:,}  dtypes={b.dtype_bytes}")
            print(f"Source:       {b.source_bytes / 1e12:.3f} TB (download)")
            print(f"Output:       {b.output_bytes / 1e12:.3f} TB ({b.export_mode})")
            print(
                f"Peak RAM:     {b.peak_ram_bytes / 1e9:.1f} GB ({b.loader}) "
                f"[eager={b.peak_ram_eager_bytes / 1e9:.1f} GB, "
                f"stream={b.peak_ram_stream_bytes / 1e9:.1f} GB]"
            )
            print(f"Weights VRAM: {b.vram_weights_bytes / 1e9:.1f} GB")
        for chk in result.checks:
            state = "OK " if chk.ok else "FAIL"
            print(
                f"  [{state}] {chk.kind}: need {chk.required_bytes / 1e9:.1f} GB, "
                f"free {chk.free_bytes / 1e9:.1f} GB"
            )
        if result.blockers:
            print("BLOCKERS:")
            for blk in result.blockers:
                print(f"  - {blk}")
        print(f"VERDICT: {'READY' if result.ok else 'REFUSED'}")

    if not result.ok:
        raise SystemExit(2)


def _cmd_info(args: argparse.Namespace) -> None:
    """Execute the 'info' subcommand."""
    from mobius.integrations.diffusers._builder import (
        _DIFFUSERS_CLASS_MAP,
        _init_diffusers_class_map,
        _load_diffusers_pipeline_index,
    )

    model_id = args.model_id

    # Check diffusers first
    pipeline_index = _load_diffusers_pipeline_index(model_id)
    if pipeline_index is not None:
        pipeline_class = pipeline_index.get("_class_name", "Unknown")
        print(f"Model:    {model_id}")
        print(f"Type:     Diffusers pipeline ({pipeline_class})")
        print("Components:")
        for comp_name, info in pipeline_index.items():
            if comp_name.startswith("_") or not isinstance(info, list):
                continue
            library, class_name = info
            _init_diffusers_class_map()
            supported = "✓" if class_name in _DIFFUSERS_CLASS_MAP else "✗"
            print(f"  {supported} {comp_name:<20} {class_name} ({library})")
        return

    # Try transformers
    import transformers

    try:
        hf_config = transformers.AutoConfig.from_pretrained(
            model_id, trust_remote_code=args.trust_remote_code
        )
    except (OSError, ValueError) as e:
        logger.debug("Failed to load config for '%s': %s", model_id, e)
        print(f"Error loading config for '{model_id}': {e}")
        return

    model_type = hf_config.model_type
    in_registry = model_type in registry

    print(f"Model:         {model_id}")
    print(f"Model type:    {model_type}")
    print(f"Supported:     {'✓' if in_registry else '✗ (not registered)'}")

    if in_registry:
        module_class = registry.get(model_type)
        task = getattr(module_class, "default_task", "text-generation")
        category = getattr(module_class, "category", "")
        print(f"Module class:  {module_class.__name__}")
        print(f"Default task:  {task}")
        print(f"Category:      {category}")

    # Show key config values
    print("Config:")
    for field in [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "vocab_size",
        "intermediate_size",
        "torch_dtype",
    ]:
        val = getattr(hf_config, field, None)
        if val is not None:
            print(f"  {field}: {val}")


def _add_release_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--release`` to a build-like subcommand.

    Shared rather than duplicated so ``build`` and ``build-gguf`` cannot drift
    into meaning different things by the same name.
    """
    parser.add_argument(
        "--release",
        action="store_true",
        help=(
            "Strip build-time debug metadata from the graph before saving. "
            "Removes the per-node provenance onnxscript records (source module "
            "path, class hierarchy, name scopes, originating rewrite rule) and "
            "symbolic-shape-inference internals — roughly 35-40%% of the "
            "serialized graph, weights excluded. Nothing reads it at inference "
            "time; keep it off while debugging a graph in Netron."
        ),
    )


class _UniqueOutputAction(argparse.Action):
    """Reject repeated ``--output`` spellings instead of silently taking the last."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str,
        option_string: str | None = None,
    ) -> None:
        if getattr(namespace, self.dest, None) is not None:
            parser.error("--output/-o may be specified only once.")
        setattr(namespace, self.dest, values)


class _MobiusArgumentParser(argparse.ArgumentParser):
    """Normalize the canonical output option and hidden positional compatibility."""

    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        parsed = super().parse_args(args, namespace)
        if getattr(parsed, "command", None) not in {"build", "build-gguf"}:
            return parsed

        output_dir = parsed.output_dir
        legacy_output_dir = parsed._legacy_output_dir
        if output_dir is None and legacy_output_dir is None:
            self.error("--output/-o is required.")
        if output_dir is not None and legacy_output_dir is not None:
            self.error("use --output/-o or the legacy positional output_dir, not both.")

        parsed.output_dir = output_dir if output_dir is not None else legacy_output_dir
        del parsed._legacy_output_dir
        return parsed


def _add_shared_build_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options that mean exactly the same thing on every build command.

    These were previously declared once per subcommand and had already drifted:
    ``--dtype`` and ``--ep`` documented the same behaviour in different words,
    which is how a real difference in behaviour eventually hides. Options whose
    semantics genuinely differ per command (``--runtime``, which ``build-gguf``
    restricts) stay declared locally, so a divergence has to be written down on
    purpose.
    """
    parser.add_argument(
        "--output",
        "-o",
        dest="output_dir",
        action=_UniqueOutputAction,
        default=None,
        metavar="OUTPUT_DIR",
        help="Output directory for the ONNX model.",
    )
    parser.add_argument(
        "_legacy_output_dir",
        nargs="?",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPE_MAP),
        default=None,
        help="Target dtype for model weights (default: f32). Weights are cast at save time.",
    )
    parser.add_argument(
        "--external-data",
        choices=["onnx", "safetensors"],
        default="onnx",
        help="External data format (default: onnx).",
    )
    parser.add_argument(
        "--max-shard-size",
        metavar="SIZE",
        default=None,
        help="Maximum external-data shard size (e.g. '5GB'). Used by both ONNX and safetensors.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        metavar="N",
        help=(
            "Number of threads used to write ONNX external data "
            "(default: 8; use 1 for serial saves)."
        ),
    )
    parser.add_argument(
        "--ep",
        "--execution-provider",
        dest="execution_provider",
        default="default",
        metavar="EP",
        help=(
            "Target execution provider for EP-aware optimizations "
            "(default: 'default' → portable ONNX, no vendor fusions). "
            "Use 'mobius list eps' to see available EPs. "
            "Examples: default, cpu, cuda, dml, webgpu, trt-rtx."
        ),
    )
    _add_release_argument(parser)


def build_parser() -> argparse.ArgumentParser:
    """Construct the full CLI parser.

    Split out of :func:`main` so the argument surface can be tested without
    running a command. It previously lived inside ``main``, which meant a test
    could only reach it by invoking ``--help`` and catching ``SystemExit`` — an
    assertion that holds regardless of what the arguments actually do.
    """
    parser = _MobiusArgumentParser(
        prog="mobius",
        description="Build ONNX models for GenAI from HuggingFace model architectures.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- build ---
    build_parser = subparsers.add_parser(
        "build",
        help="Build an ONNX model.",
        usage=(
            "mobius build (--model MODEL_ID | --config CONFIG_PATH) "
            "--output OUTPUT_DIR [options]"
        ),
    )
    source_group = build_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--model",
        metavar="MODEL_ID",
        help="HuggingFace model ID (e.g. 'meta-llama/Llama-3-8B').",
    )
    source_group.add_argument(
        "--config",
        metavar="CONFIG_PATH",
        help="Path to a local model directory containing config.json (and optionally safetensors weights).",
    )
    build_parser.add_argument(
        "--task",
        default=None,
        help="Model task (auto-detected if not specified). Use 'mobius list tasks' to see available tasks.",
    )
    build_parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Do not include weights in the output ONNX model.",
    )
    build_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading the HuggingFace model config.",
    )
    build_parser.add_argument(
        "--revision",
        default=None,
        help=(
            "Immutable HuggingFace revision used for config, weights, tokenizer, "
            "and processor assets."
        ),
    )
    build_parser.add_argument(
        "--optimize",
        nargs="?",
        const="all",
        default=None,
        metavar="RULES",
        help=(
            "Apply rewrite rules after building. "
            "Use without value for all rules, or specify comma-separated rule names "
            "(e.g. --optimize=group_query_attention,skip_norm). "
            "Available: group_query_attention, packed_attention, skip_norm."
        ),
    )
    build_parser.add_argument(
        "--component",
        default=None,
        metavar="NAME",
        help="Build only this component from a diffusers pipeline (e.g. --component vae_decoder).",
    )
    build_parser.add_argument(
        "--features",
        action="append",
        default=None,
        metavar="FEATURES",
        help=(
            "Comma-separated list of build features to enable (cargo-style; "
            "may be repeated). Available: "
            + ", ".join(sorted(_BUILD_FEATURES))
            + ". Example: --features fp8-kv-cache,static-cache. This is the "
            "canonical way to enable these build modes."
        ),
    )
    build_parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        metavar="N",
        help="Maximum sequence length for static cache buffers. "
        "Only used with --features static-cache. "
        "Defaults to max_position_embeddings from config.",
    )
    build_parser.add_argument(
        "--runtime",
        default=None,
        choices=["ort-genai", "onnx-genai"],
        metavar="RUNTIME",
        help=(
            "Generate runtime-specific config files after building. Supports: "
            "'ort-genai' (writes genai_config.json + copies tokenizer files) and "
            "'onnx-genai' (writes inference_metadata.yaml — a decoder attention/KV "
            "document for LLMs, or an iterative pipeline document for diffusion). "
            "When used with --model, tokenizer files are downloaded from HuggingFace; "
            "with --config (local directory), they are copied from that directory."
        ),
    )
    build_parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        metavar="SCALE",
        help=(
            "Classifier-free guidance scale for --runtime onnx-genai diffusion "
            "metadata. Required for conditioned diffusion pipelines so export does "
            "not guess a source pipeline's generation default; pass 1.0 explicitly "
            "for unguided generation."
        ),
    )
    build_parser.add_argument(
        "--kv-cache-scale-file",
        dest="kv_cache_scale_file",
        default=None,
        metavar="PATH",
        help=(
            "Optional JSON file of calibrated per-layer FP8 KV-cache scales "
            "(onnxruntime-genai format: {'scales': {'k_scales': [...], "
            "'v_scales': [...]}}). Only used with --features fp8-kv-cache; "
            "without it all layers use a unit scale of 1.0."
        ),
    )
    build_parser.add_argument(
        "--dequantize",
        action="store_true",
        help=(
            "Explicitly reconstruct supported compressed-tensors weights as dense "
            "floating point. By default their FP8/NVFP4 storage is preserved."
        ),
    )
    _add_shared_build_arguments(build_parser)
    build_parser.set_defaults(func=_cmd_build)

    # --- build-gguf ---
    gguf_parser = subparsers.add_parser(
        "build-gguf",
        help="Build ONNX model from a GGUF file.",
        usage="mobius build-gguf GGUF_PATH --output OUTPUT_DIR [options]",
    )
    gguf_parser.add_argument(
        "gguf_path",
        help="Path to a .gguf model file.",
    )
    gguf_parser.add_argument(
        "--mmproj",
        default=None,
        metavar="PATH",
        help=(
            "Path to a companion 'clip' mmproj GGUF (vision/audio encoder). "
            "When set, builds a full multimodal package (decoder + "
            "vision_encoder + embedding) instead of a text-only model. "
            "Supports registry-evidenced vision projectors; audio is experimental."
        ),
    )
    gguf_parser.add_argument(
        "--image-token-id",
        type=int,
        default=None,
        help=(
            "Processor-owned image placeholder ID for --mmproj packages. "
            "Use this for sentinels absent from the text vocabulary (for example -200)."
        ),
    )
    gguf_parser.add_argument(
        "--dequantize",
        action="store_true",
        help=(
            "Dequantize all mapped GGUF weights to float. Without this flag, Mobius "
            "keeps quantized target storage where supported, but may lossily normalize "
            "source qtypes and emits a quantization_report.json fidelity report."
        ),
    )
    gguf_parser.add_argument(
        "--reuse-gguf-weights",
        action="store_true",
        help=(
            "Reuse compatible tensor byte ranges directly from the original GGUF. "
            "The GGUF must be a real file in the flat output directory; converted "
            "weights are written to model.onnx.data."
        ),
    )
    gguf_parser.add_argument(
        "--target-config",
        default=None,
        metavar="PATH",
        help=(
            "Exact target model directory or config.json for a dflash/eagle3 "
            "speculative draft. The adjacent tokenizer.json is required and its "
            "ordered vocabulary must exactly match the GGUF tokenizer."
        ),
    )
    gguf_parser.add_argument(
        "--runtime",
        choices=["ort-genai", "onnx-genai"],
        default=None,
        help=(
            "Generate runtime-specific config files after building. "
            "Both routes require an exact tokenizer.huggingface.json embedded in "
            "the GGUF; opaque tokenizer.ggml.pre metadata is not reconstructed. "
            "'onnx-genai' writes inference_metadata.yaml and 'ort-genai' writes "
            "genai_config.json."
        ),
    )
    gguf_parser.add_argument(
        "--runtime-version",
        default=None,
        help=(
            "Exact selected runtime version. Required once an architecture has a "
            "runtime-supported evidence record; it must equal the version validated there."
        ),
    )
    gguf_parser.add_argument(
        "--tokenizer-repository",
        default=None,
        metavar="OWNER/REPO",
        help=(
            "Exact Hugging Face repository containing tokenizer assets for runtime "
            "packaging. Requires --tokenizer-revision and must match runtime evidence."
        ),
    )
    gguf_parser.add_argument(
        "--tokenizer-revision",
        default=None,
        metavar="COMMIT_SHA",
        help="Immutable 40-hex revision for --tokenizer-repository.",
    )
    gguf_parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only already-cached pinned tokenizer assets; perform no Hub requests.",
    )
    gguf_parser.add_argument(
        "--static-cache",
        action="store_true",
        help=(
            "Use a static KV cache (pre-allocated fixed-width buffers written "
            "in place via TensorScatter) instead of the dynamic concat-grow "
            "cache. Produces a fully static-shaped graph as required by "
            "fixed-shape runtimes such as the QNN HTP backend."
        ),
    )
    gguf_parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Maximum sequence length for static cache buffers. Only used with "
            "--static-cache. Defaults to max_position_embeddings from config."
        ),
    )
    _add_shared_build_arguments(gguf_parser)
    gguf_parser.set_defaults(func=_cmd_build_gguf)

    # --- list ---
    list_parser = subparsers.add_parser(
        "list", help="List supported models, tasks, dtypes, or EPs."
    )
    list_parser.add_argument(
        "resource",
        choices=["models", "tasks", "dtypes", "eps"],
        help="What to list.",
    )
    list_parser.set_defaults(func=_cmd_list)

    # --- info ---
    info_parser = subparsers.add_parser("info", help="Show information about a model.")
    info_parser.add_argument(
        "model_id",
        help="HuggingFace model ID to inspect.",
    )
    info_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading the HuggingFace model config.",
    )
    info_parser.set_defaults(func=_cmd_info)

    # --- preflight ---
    preflight_parser = subparsers.add_parser(
        "preflight",
        help="Dry-run an export: validate shard metadata and compute the exact "
        "disk/RAM/VRAM budget, refusing before any download if it will not fit.",
    )
    preflight_parser.add_argument(
        "model_id",
        help="HuggingFace model ID or local checkpoint directory to preflight.",
    )
    preflight_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Intended output directory for the exported ONNX (checked for free space).",
    )
    preflight_parser.add_argument(
        "--revision",
        default=None,
        help="Model revision/branch/commit to resolve (default: main).",
    )
    preflight_parser.add_argument(
        "--download-dir",
        default=None,
        help="Filesystem checked for the source-download budget. Default: the "
        "Hugging Face cache (HF_HUB_CACHE), where hf_hub_download actually "
        "writes shards. Set this (or HF_HOME) to a large volume to relocate the "
        "download.",
    )
    preflight_parser.add_argument(
        "--export-mode",
        choices=["passthrough", "fp16", "int4-qmoe"],
        default="passthrough",
        help="Weight representation of the exported artifact (default: passthrough).",
    )
    preflight_parser.add_argument(
        "--loader",
        choices=["eager", "stream"],
        default="stream",
        help="Weight application strategy that sets the host-RAM peak.",
    )
    preflight_parser.add_argument(
        "--group-size",
        type=int,
        default=32,
        help="Quantization block size for int4-qmoe output sizing (default: 32).",
    )
    preflight_parser.add_argument(
        "--target-dtype-bytes",
        type=float,
        default=None,
        help="Bytes/param of the resident runtime weights for the VRAM estimate. "
        "Default: derived from the export (int4-qmoe ~0.5, fp16/passthrough 2.0). "
        "Override to model a runtime dtype that differs from the export.",
    )
    preflight_parser.add_argument(
        "--gpu-total-bytes",
        default=None,
        help="Largest single-GPU memory (e.g. '80GB'); default probes nvidia-smi.",
    )
    preflight_parser.add_argument(
        "--no-gpu-probe",
        action="store_true",
        help="Do not probe nvidia-smi for GPU memory.",
    )
    preflight_parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Fractional free-space headroom required above each budget (default 0.05).",
    )
    preflight_parser.add_argument(
        "--state",
        default=None,
        help="Resumable state JSON file; records validated shards and refuses on "
        "checkpoint identity drift.",
    )
    preflight_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full verdict as JSON.",
    )
    preflight_parser.set_defaults(func=_cmd_preflight)

    # --- convert-comfyui ---
    comfy_parser = subparsers.add_parser(
        "convert-comfyui",
        help="Translate a ComfyUI API-format workflow JSON into an onnx-genai "
        "pipeline metadata directory (inference_metadata.yaml + run.json). The "
        "ONNX component graphs are built separately by Mobius's diffusers builder.",
    )
    comfy_parser.add_argument("workflow", help="Path to the ComfyUI API-format workflow JSON.")
    comfy_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional diffusers directory or Hugging Face model id whose "
        "scheduler config supplies the noise-schedule betas (Stable Diffusion "
        "defaults are used when omitted).",
    )
    comfy_parser.add_argument(
        "--revision",
        default=None,
        help="Pinned Hugging Face revision used to resolve the checkpoint scheduler config.",
    )
    comfy_parser.add_argument(
        "--output", "-o", required=True, help="Output directory for the pipeline metadata."
    )
    comfy_parser.add_argument(
        "--sdxl",
        action="store_true",
        help="Target an SDXL pipeline (routes the dual text-encoder conditioning edges).",
    )
    comfy_parser.set_defaults(func=_cmd_convert_comfyui)

    # --- preflight-gguf ---
    preflight_parser = subparsers.add_parser(
        "preflight-gguf",
        help="Metadata-only preflight of a GGUF file or split set (local path "
        "or Hugging Face 'owner/repo[:file]'). Reports exact files, bytes, "
        "checksums, resolved architecture, and export blockers (notably the "
        "sparse-MoE fusion blocker) WITHOUT downloading tensor payloads.",
    )
    preflight_parser.add_argument(
        "source",
        help="Local .gguf path / split-set shard / directory, or a Hugging "
        "Face reference 'owner/repo' or 'owner/repo:filename.gguf'.",
    )
    preflight_parser.add_argument(
        "--filename",
        default=None,
        help="Specific shard filename when SOURCE is a bare 'owner/repo'.",
    )
    preflight_parser.add_argument(
        "--revision",
        default=None,
        help="Pinned Hugging Face revision (sha/branch/tag).",
    )
    preflight_parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Compute per-shard sha256 for a local set (reads file bytes, not "
        "tensor payloads via the model API).",
    )
    preflight_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of the human-readable summary.",
    )
    preflight_parser.add_argument(
        "--cache",
        default=None,
        help="Optional JSON cache path; a resumable, idempotent preflight reads "
        "from it when present and writes to it otherwise.",
    )
    preflight_parser.set_defaults(func=_cmd_preflight_gguf)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the CLI."""
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
