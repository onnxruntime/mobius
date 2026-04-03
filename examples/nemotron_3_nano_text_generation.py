#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""NemotronH text generation — standalone greedy decoding (no onnxruntime-genai).

Demonstrates a fully manual autoregressive generation loop for NemotronH
(nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16) with the **hybrid Mamba2 + Attention
+ MLP** architecture.

NemotronH models use a mix of:
- **Mamba2 layers**: conv_state + ssm_state (SSM carry tensors).
- **Full-attention layers**: standard KV cache (key + value).
- **MLP layers**: stateless (no cache).

This example handles all three layer types in the decoding loop.
Because Mamba2 layers require sequential processing, every token
(including the prompt) is processed one at a time.

The model is instruction-tuned and uses a ChatML template with a
``<think>...</think>`` reasoning format.

Usage::

    # Interactive mode (no --prompt → prompts you for queries):
    python examples/nemotron_3_nano_text_generation.py --device cuda

    # Single prompt:
    python examples/nemotron_3_nano_text_generation.py --prompt "What is 2+3?"

    # Interactive mode with HF comparison (runs HF after all queries):
    python examples/nemotron_3_nano_text_generation.py --device cuda --compare-hf

    # Run on GPU:
    python examples/nemotron_3_nano_text_generation.py --device cuda

    # Save the ONNX model to disk without running inference:
    python examples/nemotron_3_nano_text_generation.py --save-to output/nemotron_h/

    # Build with float16 precision:
    python examples/nemotron_3_nano_text_generation.py --dtype f16
"""

from __future__ import annotations

import argparse
import os
import re
import time

import ml_dtypes
import numpy as np
import transformers

from mobius import build
from mobius._flags import override_flags
from mobius._testing.ort_inference import OnnxModelSession

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
DEFAULT_PROMPT = "What is the capital of France?"
MAX_NEW_TOKENS = 300

DTYPE_MAP = {"f16": np.float16, "f32": np.float32, "bf16": ml_dtypes.bfloat16}


# ---------------------------------------------------------------------------
# NemotronH monkey-patch for HuggingFace transformers
# ---------------------------------------------------------------------------


def _fix_nemotron_h_dt_bias(model) -> None:
    """Reload dt_bias and out_proj.weight from the checkpoint.

    HuggingFace's ``_init_weights`` unconditionally overwrites these
    parameters *after* loading checkpoint weights:

    - ``dt_bias``: reinitialized with ``torch.rand(...)``
    - ``out_proj.weight``: reinitialized with ``kaiming_uniform_`` then
      scaled by ``1/sqrt(num_hidden_layers)``
      (when ``config.rescale_prenorm_residual`` is True)

    This reloads the correct values directly from safetensors.
    """
    import torch
    from safetensors.torch import load_file

    from huggingface_hub import HfApi

    # Keys that _init_weights corrupts and must be reloaded
    _CORRUPTED_KEYS = {"dt_bias", "out_proj.weight"}

    model_id = model.config._name_or_path
    api = HfApi()
    shard_files = [
        f.rfilename
        for f in api.model_info(model_id).siblings
        if f.rfilename.endswith(".safetensors")
    ]
    repo_dir = os.path.dirname(
        transformers.utils.cached_file(model_id, shard_files[0])
    )

    sd = model.state_dict()

    patched = 0
    for shard_name in shard_files:
        shard_path = os.path.join(repo_dir, shard_name)
        shard_weights = load_file(shard_path)
        for key, value in shard_weights.items():
            if not any(c in key for c in _CORRUPTED_KEYS):
                continue
            # Try direct match first, then backbone. <-> model. remap
            sd_key = key
            if sd_key not in sd:
                if key.startswith("backbone."):
                    sd_key = "model." + key[len("backbone."):]
                elif key.startswith("model."):
                    sd_key = "backbone." + key[len("model."):]
            if sd_key in sd:
                sd[sd_key] = value
                patched += 1
    if patched:
        model.load_state_dict(sd)
        print(f"  Fixed {patched} corrupted parameters from checkpoint")


def _apply_nemotron_h_patch():
    """Monkey-patch HuggingFace NemotronH bugs.

    Fixes three issues in transformers' NemotronH implementation:
    1. _pattern_to_list: missing "-" -> "mlp" mapping
    2. _validate_layers_block_type: "mlp" not in valid types
    3. NemotronHModel.forward: block_type_to_mask missing "mlp" key
    """
    try:
        from transformers.models.nemotron_h.configuration_nemotron_h import (
            NemotronHConfig,
        )

        @staticmethod
        def _patched_pattern_to_list(pattern: str) -> list:
            mapping = {"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}
            return [mapping[char] for char in pattern]

        @staticmethod
        def _patched_list_to_pattern(layers_list: list) -> str:
            reverse = {"mamba": "M", "moe": "E", "attention": "*", "mlp": "-"}
            return "".join(reverse[t] for t in layers_list)

        @staticmethod
        def _patched_validate(
            layers_block_type,
            expected_length=None,
            param_name="layers_block_type",
        ):
            if not isinstance(layers_block_type, list):
                raise TypeError(f"{param_name} must be a list of strings.")
            if expected_length is not None and len(layers_block_type) != expected_length:
                raise ValueError(f"{param_name} must have length {expected_length}.")
            valid_types = {"mamba", "attention", "moe", "mlp"}
            invalid = set(layers_block_type) - valid_types
            if invalid:
                raise ValueError(f"{param_name} contains invalid types: {invalid}.")

        NemotronHConfig._pattern_to_list = _patched_pattern_to_list
        NemotronHConfig._list_to_pattern = _patched_list_to_pattern
        NemotronHConfig._validate_layers_block_type = _patched_validate

        import functools

        from transformers.models.nemotron_h import modeling_nemotron_h
        from transformers.models.nemotron_h.modeling_nemotron_h import (
            NemotronHMLP,
        )

        if "mlp" not in modeling_nemotron_h.MIXER_TYPES:
            modeling_nemotron_h.MIXER_TYPES["mlp"] = lambda config, layer_idx=None: (
                NemotronHMLP(config)
            )

        orig_nm_forward = modeling_nemotron_h.NemotronHModel.forward
        if getattr(orig_nm_forward, "_nemotron_h_patched", False):
            return  # already patched

        @functools.wraps(orig_nm_forward)
        def _nm_forward_with_mlp(self, *args, **kwargs):
            mlp_layers = [
                layer for layer in self.layers if getattr(layer, "block_type", None) == "mlp"
            ]
            for layer in mlp_layers:
                layer.block_type = "moe"
            try:
                return orig_nm_forward(self, *args, **kwargs)
            finally:
                for layer in mlp_layers:
                    layer.block_type = "mlp"

        _nm_forward_with_mlp._nemotron_h_patched = True
        modeling_nemotron_h.NemotronHModel.forward = _nm_forward_with_mlp
        print("Applied NemotronH monkey-patches")
    except ImportError:
        print("NemotronH modules not available; skipping monkey-patches")


# ---------------------------------------------------------------------------
# Hybrid state initialization
# ---------------------------------------------------------------------------


def init_hybrid_states(config, dtype: np.dtype = np.float32) -> dict[str, np.ndarray]:
    """Initialize per-layer states for the hybrid architecture.

    Mamba2 layers get zero-filled conv_state and ssm_state.
    Full-attention layers get empty KV caches (past_seq_len=0).
    MLP layers have no state.
    """
    batch_size = 1
    states: dict[str, np.ndarray] = {}
    layer_types = getattr(config, "layers_block_type", getattr(config, "layer_types", None)) or []

    # Mamba2 dims from NemotronH config (handle both old and new attr names)
    n_heads = getattr(config, "mamba_n_heads", None) or config.mamba_num_heads
    d_head = getattr(config, "mamba_d_head", None) or config.mamba_head_dim
    d_state = getattr(config, "mamba_d_state", None) or config.ssm_state_size
    n_groups = getattr(config, "mamba_n_groups", None) or config.n_groups
    d_inner = n_heads * d_head
    d_conv = getattr(config, "mamba_d_conv", None) or config.conv_kernel
    # conv_state covers d_inner + 2 * n_groups * d_state
    conv_dim = d_inner + 2 * n_groups * d_state

    for i in range(config.num_hidden_layers):
        ltype = layer_types[i] if i < len(layer_types) else "full_attention"

        if ltype in ("mamba", "mamba2"):
            states[f"past_key_values.{i}.conv_state"] = np.zeros(
                (batch_size, conv_dim, d_conv - 1), dtype=dtype
            )
            # ssm_state: (batch, n_heads, d_head, d_state)
            states[f"past_key_values.{i}.ssm_state"] = np.zeros(
                (batch_size, n_heads, d_head, d_state),
                dtype=dtype,
            )
        elif ltype in ("attention", "full_attention"):
            states[f"past_key_values.{i}.key"] = np.zeros(
                (
                    batch_size,
                    config.num_key_value_heads,
                    0,
                    config.head_dim,
                ),
                dtype=dtype,
            )
            states[f"past_key_values.{i}.value"] = np.zeros(
                (
                    batch_size,
                    config.num_key_value_heads,
                    0,
                    config.head_dim,
                ),
                dtype=dtype,
            )
        # MLP layers have no state

    return states


def update_states(
    states: dict[str, np.ndarray],
    outputs: dict[str, np.ndarray],
    config,
) -> dict[str, np.ndarray]:
    """Copy present-state outputs back into the past-state inputs."""
    layer_types = getattr(config, "layers_block_type", getattr(config, "layer_types", None)) or []
    new_states: dict[str, np.ndarray] = {}

    for i in range(config.num_hidden_layers):
        ltype = layer_types[i] if i < len(layer_types) else "full_attention"

        if ltype in ("mamba", "mamba2"):
            new_states[f"past_key_values.{i}.conv_state"] = outputs[f"present.{i}.conv_state"]
            new_states[f"past_key_values.{i}.ssm_state"] = outputs[f"present.{i}.ssm_state"]
        elif ltype in ("attention", "full_attention"):
            new_states[f"past_key_values.{i}.key"] = outputs[f"present.{i}.key"]
            new_states[f"past_key_values.{i}.value"] = outputs[f"present.{i}.value"]

    return new_states


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def tokenize_prompt(tokenizer, prompt: str, use_chat: bool) -> list[int]:
    """Tokenize a prompt, optionally applying the chat template."""
    if use_chat:
        messages = [{"role": "user", "content": prompt}]
        result = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if isinstance(result, dict):
            input_ids = result["input_ids"]
        elif hasattr(result, "input_ids"):
            input_ids = result.input_ids
        else:
            input_ids = result
        return list(input_ids)
    return tokenizer.encode(prompt)


def parse_think_output(text: str) -> tuple[str, str]:
    """Split model output into (reasoning, answer) from <think>...</think>."""
    match = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    if text.startswith("<think>"):
        return text[len("<think>") :].strip(), ""
    return "", text.strip()


def format_output(tokenizer, generated_ids: list[int]) -> str:
    """Decode generated tokens and extract the answer."""
    text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    for special in ["<|im_end|>", "</s>", "<|endoftext|>"]:
        text = text.replace(special, "")
    return text.strip()


def display_output(tokenizer, generated_ids: list[int]) -> None:
    """Decode, strip special tokens, and display with think/answer parsing."""
    text = format_output(tokenizer, generated_ids)
    reasoning, answer = parse_think_output(text)
    if reasoning:
        print(f"  Thinking: {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}")
    if answer:
        print(f"  Answer: {answer}")
    elif not reasoning:
        print(f"  Output: {text[:200]}")
    print()


# ---------------------------------------------------------------------------
# ONNX generation
# ---------------------------------------------------------------------------


def generate(
    session: OnnxModelSession,
    tokenizer,
    prompt: str,
    config,
    *,
    dtype: np.dtype = np.float32,
    max_new_tokens: int = MAX_NEW_TOKENS,
    use_chat: bool = True,
) -> str:
    """Greedy autoregressive generation with the hybrid architecture.

    Uses a single ONNX model built with ``NemotronHCausalLMModel``.
    Because Mamba2 layers only support single-token decode, every
    token (including the prompt) is processed one at a time.
    """
    input_ids = tokenize_prompt(tokenizer, prompt, use_chat)
    batch_size = 1
    prompt_len = len(input_ids)

    if prompt_len == 0:
        raise ValueError("Tokenized prompt is empty; cannot generate.")

    states = init_hybrid_states(config, dtype=dtype)
    past_seq_len = 0
    generated_ids: list[int] = []

    t0 = time.time()

    # Process prompt tokens one at a time (Mamba2 requires seq_len=1)
    for t in range(prompt_len):
        cur_token = np.array([[input_ids[t]]], dtype=np.int64)
        total_seq_len = past_seq_len + 1

        feeds = {
            "input_ids": cur_token,
            "attention_mask": np.ones((batch_size, total_seq_len), dtype=np.int64),
            "position_ids": np.array([[past_seq_len]], dtype=np.int64),
            **states,
        }

        outputs = session.run(feeds)
        states = update_states(states, outputs, config)
        past_seq_len = total_seq_len

    prefill_time = time.time() - t0
    print(f"  Prefill: {prompt_len} tokens in {prefill_time:.2f}s")

    # Generate new tokens
    logits = outputs["logits"]
    next_token_id = int(np.argmax(logits[:, -1, :]))
    t0 = time.time()

    for _ in range(max_new_tokens):
        generated_ids.append(next_token_id)

        if next_token_id == tokenizer.eos_token_id:
            break

        cur_input_ids = np.array([[next_token_id]], dtype=np.int64)
        total_seq_len = past_seq_len + 1

        feeds = {
            "input_ids": cur_input_ids,
            "attention_mask": np.ones((batch_size, total_seq_len), dtype=np.int64),
            "position_ids": np.array([[past_seq_len]], dtype=np.int64),
            **states,
        }

        outputs = session.run(feeds)
        states = update_states(states, outputs, config)
        past_seq_len = total_seq_len

        logits = outputs["logits"]
        next_token_id = int(np.argmax(logits[:, -1, :]))

    gen_time = time.time() - t0
    if generated_ids:
        tps = len(generated_ids) / gen_time if gen_time > 0 else 0
        print(f"  Generated: {len(generated_ids)} tokens in {gen_time:.1f}s ({tps:.1f} tok/s)")

    display_output(tokenizer, generated_ids)
    return format_output(tokenizer, generated_ids)


# ---------------------------------------------------------------------------
# HuggingFace comparison
# ---------------------------------------------------------------------------


def generate_hf(
    model_id: str,
    prompt: str,
    tokenizer,
    max_new_tokens: int,
    device: str = "cpu",
    use_chat: bool = True,
) -> str:
    """Run text-only generation with HuggingFace transformers."""
    import torch

    _apply_nemotron_h_patch()

    print(f"[HF] Loading {model_id} ...")
    torch_dtype = torch.float32 if device == "cpu" else "auto"
    model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=None
        )
        .to(device)
        .eval()
    )
    _fix_nemotron_h_dt_bias(model)

    input_ids = tokenize_prompt(tokenizer, prompt, use_chat)
    ids_tensor = torch.tensor([input_ids], device=device)

    print(f"[HF] Prompt: {prompt}")
    print("-" * 40)

    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            ids_tensor,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0
    gen_ids = output[0, len(input_ids) :].tolist()
    print(f"  Generated: {len(gen_ids)} tokens in {elapsed:.1f}s")

    display_output(tokenizer, gen_ids)
    text = format_output(tokenizer, gen_ids)
    print("-" * 40)
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "NemotronH text generation — standalone greedy decoding (no onnxruntime-genai)."
        ),
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help="HuggingFace model ID (default: %(default)s).",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help=(
            "Text prompt. If omitted, enters interactive mode "
            "(type queries, 'quit' or Ctrl-D to stop)."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Maximum tokens to generate (default: %(default)s).",
    )
    parser.add_argument(
        "--save-to",
        metavar="DIR",
        default=None,
        help="Save the ONNX model package to DIR and exit (no inference).",
    )
    parser.add_argument(
        "--load-from",
        metavar="DIR",
        default=None,
        help="Load a previously saved ONNX model from DIR (skip build).",
    )
    parser.add_argument(
        "--dtype",
        default="f32",
        choices=["f16", "bf16", "f32"],
        help="Precision type for the ONNX model (default: %(default)s).",
    )
    parser.add_argument(
        "--compare-hf",
        action="store_true",
        help="Also run with HuggingFace transformers and compare outputs.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for ONNX Runtime and PyTorch inference (default: %(default)s).",
    )
    parser.add_argument(
        "--no-chat",
        action="store_true",
        help="Disable chat template (send raw text).",
    )
    args = parser.parse_args()

    use_chat = not args.no_chat

    if args.load_from:
        import onnx_ir as ir

        from mobius._configs import NemotronHConfig

        print(f"Loading ONNX model from {args.load_from} ...")
        onnx_model = ir.load(os.path.join(args.load_from, "model.onnx"))
        hf_config = transformers.AutoConfig.from_pretrained(
            args.model,
            trust_remote_code=True,
        )
        config = NemotronHConfig.from_transformers(hf_config)
        session = OnnxModelSession(onnx_model, device=args.device)
    else:
        build_flags = {}
        if args.device == "cuda":
            build_flags["ort_cuda_grouped_rmsnorm_workaround"] = True
        print(f"Building model for {args.model!r} (dtype={args.dtype}) ...")
        with override_flags(**build_flags):
            pkg = build(
                args.model,
                dtype=args.dtype,
                load_weights=True,
                trust_remote_code=True,
            )
        config = pkg.config

        if args.save_to:
            pkg.save(args.save_to, external_data="onnx")
            print(f"Saved to {args.save_to}")
            return

        session = OnnxModelSession(pkg["model"], device=args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

    # Collect prompts: either single --prompt or interactive loop
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = []
        print("\nInteractive mode — enter prompts (type 'quit' or Ctrl-D to stop).")

    # Run ONNX generation for each prompt
    results: list[tuple[str, str]] = []  # (prompt, onnx_output)
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        if use_chat:
            print("(chat template enabled)")
        print("-" * 40)
        onnx_output = generate(
            session, tokenizer, prompt, config,
            dtype=DTYPE_MAP[args.dtype],
            max_new_tokens=args.max_new_tokens,
            use_chat=use_chat,
        )
        print("-" * 40)
        results.append((prompt, onnx_output))

    if not args.prompt:
        # Interactive loop
        while True:
            try:
                prompt = input("\n> ").strip()
            except EOFError:
                print()
                break
            if not prompt or prompt.lower() == "quit":
                break
            print(f"\nPrompt: {prompt}")
            if use_chat:
                print("(chat template enabled)")
            print("-" * 40)
            onnx_output = generate(
                session, tokenizer, prompt, config,
                dtype=DTYPE_MAP[args.dtype],
                max_new_tokens=args.max_new_tokens,
                use_chat=use_chat,
            )
            print("-" * 40)
            results.append((prompt, onnx_output))

    if args.compare_hf and results:
        print("\n" + "=" * 60)
        print("HuggingFace Transformers Comparison")
        print("=" * 60)
        for prompt, onnx_output in results:
            hf_output = generate_hf(
                args.model, prompt, tokenizer,
                args.max_new_tokens,
                device=args.device,
                use_chat=use_chat,
            )
            _, answer_onnx = parse_think_output(onnx_output)
            _, answer_hf = parse_think_output(hf_output)

            print(f"\nPrompt: {prompt!r}")
            if answer_onnx == answer_hf:
                print("  ✓ Answers match exactly!")
            else:
                print("  ✗ Answers differ:")
                print(f"    ONNX: {answer_onnx!r}")
                print(f"    HF:   {answer_hf!r}")


if __name__ == "__main__":
    main()
