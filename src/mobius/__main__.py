# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Command-line interface for mobius."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from typing import TYPE_CHECKING

import tqdm

if TYPE_CHECKING:
    import onnx_ir as ir
    import torch

from mobius._builder import (
    DTYPE_MAP,
    build_from_module,
    resolve_dtype,
)
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
        """Create the correct static cache task for the given model type."""
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
    trust_remote_code = args.trust_remote_code
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
        pipeline_index = _load_diffusers_pipeline_index(args.model)
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
            dtype=dtype_override,
            execution_provider=execution_provider,
        )
        _save_package(pkg, output_dir, args, optimize, component_filter)
        return

    # Build from HuggingFace model ID or local config
    if args.config:
        import transformers

        config_path = args.config
        hf_config = transformers.AutoConfig.from_pretrained(
            config_path, trust_remote_code=trust_remote_code
        )
        model_type = hf_config.model_type
        parent_config = hf_config
        if hasattr(hf_config, "text_config"):
            hf_config = hf_config.text_config
        config = _config_from_hf(hf_config, parent_config=parent_config)
        if dtype_override is not None:
            config = dataclasses.replace(config, dtype=dtype_override)
        if static_cache_params is not None:
            task = _resolve_static_cache_task(model_type)
        elif task is None:
            task = _default_task_for_model(model_type)
        module_class = registry.get(model_type)
        model_module = module_class(config)
        pkg = build_from_module(
            model_module,
            config,
            task=task,
            execution_provider=execution_provider,
            fp8_kv_cache=fp8_kv_cache,
            kv_cache_scales=kv_cache_scales,
            prune_prefill_prefix=prune_prefill_prefix,
        )
        for name, model in pkg.items():
            model.graph.name = f"{config_path}/{name}"
        if load_weights:
            state_dict = _load_weights_from_dir(config_path)
            if hasattr(model_module, "preprocess_weights"):
                state_dict = model_module.preprocess_weights(state_dict)
            pkg.apply_weights(state_dict)
    else:
        model_id_or_path = args.model
        if static_cache_params is not None:
            # Detect model type to resolve the correct static cache task.
            import transformers

            hf_config = transformers.AutoConfig.from_pretrained(
                model_id_or_path, trust_remote_code=trust_remote_code
            )
            task = _resolve_static_cache_task(getattr(hf_config, "model_type", ""))

        pkg = build(
            model_id_or_path,
            task=task,
            dtype=dtype_override,
            load_weights=load_weights,
            trust_remote_code=trust_remote_code,
            execution_provider=execution_provider,
            text_only=args.text_only,
            fp8_kv_cache=fp8_kv_cache,
            kv_cache_scales=kv_cache_scales,
            prune_prefill_prefix=prune_prefill_prefix,
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
        if is_native_vlm_package(pkg):
            try:
                artifacts = write_native_vlm_package_metadata(
                    pkg,
                    output_dir,
                    config=config,
                    source=source,
                )
            except ValueError as error:
                raise SystemExit(f"Error: {error}") from error
        else:
            artifacts = write_onnx_genai_config(pkg, output_dir, config=config, source=source)
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

    if getattr(args, "runtime", None) == "ort-genai":
        raise SystemExit(
            "Error: mobius build-gguf does not yet support --runtime ort-genai. "
            "The command cannot emit a valid genai_config.json until the selected "
            "GGUF architecture's cache and tokenizer contracts have passed real "
            "ORT GenAI generation. Use --runtime onnx-genai where supported, or "
            "omit --runtime and run the ONNX model directly."
        )

    mmproj_path = getattr(args, "mmproj", None)
    keep_quantized = not args.dequantize

    if keep_quantized:
        print("Preserving supported GGUF quantization (float-only inputs stay float)...")
    else:
        print("Dequantized mode: converting GGUF weights to float...")

    gguf_path = args.gguf_path
    output_dir = args.output or os.path.splitext(gguf_path)[0] + "_onnx"
    os.makedirs(output_dir, exist_ok=True)

    if args.max_seq_len is not None and not args.static_cache:
        raise SystemExit("Error: --max-seq-len can only be used with --static-cache.")
    if args.max_seq_len is not None and args.max_seq_len <= 0:
        raise SystemExit("Error: --max-seq-len must be a positive integer.")
    if args.max_workers <= 0:
        raise SystemExit("Error: --max-workers must be a positive integer.")
    if mmproj_path is not None and args.static_cache:
        raise SystemExit("Error: --static-cache cannot be used with --mmproj.")

    pkg = build_from_gguf(
        gguf_path,
        mmproj=mmproj_path,
        dtype=args.dtype,
        keep_quantized=keep_quantized,
        execution_provider=args.execution_provider,
        static_cache=args.static_cache,
        max_seq_len=args.max_seq_len,
    )

    pkg.save(
        output_dir,
        external_data=args.external_data,
        max_workers=args.max_workers,
    )
    for name in pkg:
        use_subfolders = len(pkg) > 1
        if use_subfolders:
            path = os.path.join(output_dir, name, "model.onnx")
        else:
            path = os.path.join(output_dir, "model.onnx")
        print(f"Saved {name} to {path}")

    if getattr(args, "runtime", None) == "onnx-genai":
        from mobius.integrations.gguf import write_gguf_tokenizer_json
        from mobius.integrations.onnx_genai import write_onnx_genai_config

        # A GGUF checkpoint has no Hugging Face source directory, so the
        # tokenizer is reconstructed from the file's embedded ggml metadata
        # rather than copied from a `source`.
        tokenizer_path = write_gguf_tokenizer_json(gguf_path, output_dir)
        if tokenizer_path is not None:
            print(f"  tokenizer: {tokenizer_path}")
        artifacts = write_onnx_genai_config(
            pkg, output_dir, config=getattr(pkg, "config", None), source=None
        )
        for name, path in artifacts.items():
            print(f"  {name}: {path}")


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


def main(argv: list[str] | None = None) -> None:
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="mobius",
        description="Build ONNX models for GenAI from HuggingFace model architectures.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- build ---
    build_parser = subparsers.add_parser("build", help="Build an ONNX model.")
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
        "output_dir",
        help="Output directory for the ONNX model.",
    )
    build_parser.add_argument(
        "--task",
        default=None,
        help="Model task (auto-detected if not specified). Use 'mobius list tasks' to see available tasks.",
    )
    build_parser.add_argument(
        "--external-data",
        choices=["onnx", "safetensors"],
        default="onnx",
        help="External data format (default: onnx).",
    )
    build_parser.add_argument(
        "--max-shard-size",
        metavar="SIZE",
        default=None,
        help="Maximum external-data shard size (e.g. '5GB'). Used by both ONNX and safetensors.",
    )
    build_parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        metavar="N",
        help="Number of threads used to write ONNX external data (default: 8; use 1 for serial saves).",
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
        "--dtype",
        choices=sorted(DTYPE_MAP),
        default=None,
        help="Target dtype for model weights (default: f32). Weights are cast at save time.",
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
    build_parser.set_defaults(func=_cmd_build)

    # --- build-gguf ---
    gguf_parser = subparsers.add_parser(
        "build-gguf", help="Build ONNX model from a GGUF file."
    )
    gguf_parser.add_argument(
        "gguf_path",
        help="Path to a .gguf model file.",
    )
    gguf_parser.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="DIR",
        help="Output directory (default: <gguf_stem>_onnx/).",
    )
    gguf_parser.add_argument(
        "--mmproj",
        default=None,
        metavar="PATH",
        help=(
            "Path to a companion 'clip' mmproj GGUF (vision/audio encoder). "
            "When set, builds a full multimodal package (decoder + "
            "vision_encoder + embedding) instead of a text-only model. "
            "Currently supports Gemma4 and Muse Glimmer vision; audio is "
            "experimental."
        ),
    )
    quantization_group = gguf_parser.add_mutually_exclusive_group()
    quantization_group.add_argument(
        "--dequantize",
        action="store_true",
        help="Dequantize all GGUF weights to float instead of preserving quantization.",
    )
    quantization_group.add_argument(
        "--keep-quantized",
        action="store_true",
        help=(
            "Deprecated compatibility alias; supported GGUF quantization is "
            "preserved by default."
        ),
    )
    gguf_parser.add_argument(
        "--dtype",
        choices=sorted(DTYPE_MAP),
        default=None,
        help="Target dtype for model weights.",
    )
    gguf_parser.add_argument(
        "--external-data",
        choices=["onnx", "safetensors"],
        default="onnx",
        help="External data format (default: onnx).",
    )
    gguf_parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        metavar="N",
        help="Number of threads used to write ONNX external data (default: 8; use 1 for serial saves).",
    )
    gguf_parser.add_argument(
        "--ep",
        "--execution-provider",
        dest="execution_provider",
        default="default",
        metavar="EP",
        help=(
            "Target execution provider for EP-aware graph optimisations "
            "(e.g. 'cpu' to apply the GroupQueryAttention rewrite). "
            "Defaults to 'default' (portable ONNX, no vendor fusions)."
        ),
    )
    gguf_parser.add_argument(
        "--runtime",
        choices=["ort-genai", "onnx-genai"],
        default=None,
        help=(
            "Generate runtime-specific config files after building. "
            "'onnx-genai' writes inference_metadata.yaml plus a tokenizer.json "
            "reconstructed from the GGUF's embedded tokenizer metadata; "
            "'ort-genai' is currently rejected until GGUF cache/tokenizer "
            "contracts have runtime generation coverage."
        ),
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
        "--output", "-o", required=True, help="Output directory for the pipeline metadata."
    )
    comfy_parser.add_argument(
        "--sdxl",
        action="store_true",
        help="Target an SDXL pipeline (routes the dual text-encoder conditioning edges).",
    )
    comfy_parser.set_defaults(func=_cmd_convert_comfyui)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
