"""Run batch-one greedy generation with a TensorRT heads-first static-cache engine.

Read generate() first for the token loop. Independent diagnostic modules live
in tensorrt_debug/: full_prefix, inspect_cache, inspect_attention, comparison.
CUDA and TensorRT details live in tensorrt_debug/runtime.py.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from tensorrt_debug.comparison import HuggingFaceReference, compare_logits
from tensorrt_debug.full_prefix import build_full_prefix, validate_full_prefix_budget
from tensorrt_debug.inspect_cache import CacheInspector
from tensorrt_debug.runtime import TensorRTRunner
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--sdk", type=Path, default=Path("C:/TensorRT-11.3.0.99"))
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", default="What is the capital of France? Answer briefly.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--compare-hf",
        action="store_true",
        help="Compare identical-token steps against HuggingFace on CPU",
    )
    parser.add_argument(
        "--compare-steps",
        type=int,
        default=2,
        help="Number of steps to compare, including prefill (default: 2)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="HF checkpoint revision; must match the engine's source weights",
    )
    parser.add_argument(
        "--full-prefix",
        action="store_true",
        help="Recompute the entire prefix with zeroed caches at every step",
    )

    parser.add_argument(
        "--inspect-cache",
        action="store_true",
        help="Snapshot cache before/after prefill and first decode; requires --compare-hf",
    )
    parser.add_argument(
        "--cache-layer", type=int, default=0, help="Layer to inspect (default: 0)"
    )
    parser.add_argument(
        "--inspect-attention",
        action="store_true",
        help="Compare layer-0 intermediates from a diagnostic engine with HF",
    )

    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.compare_hf and not 2 <= args.compare_steps <= args.max_new_tokens:
        parser.error("--compare-steps must be between 2 and --max-new-tokens")
    if args.inspect_cache and (not args.compare_hf or args.full_prefix):
        parser.error(
            "--inspect-cache requires --compare-hf and cached mode (no --full-prefix)"
        )
    if args.cache_layer < 0:
        parser.error("--cache-layer must be nonnegative")
    if args.inspect_attention and not args.compare_hf:
        parser.error("--inspect-attention requires --compare-hf")

    return args


def run(args, runner):
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    current_ids = np.asarray(
        [tokenizer.encode(rendered, add_special_tokens=False)], dtype=np.int64
    )
    prompt_ids = current_ids.copy()
    if args.inspect_attention and not {"probe.query", "probe.attention"}.issubset(
        runner.names
    ):
        raise ValueError("Use an engine built by tensorrt_attention_probe.py")
    cache_inspector = CacheInspector(runner, args.cache_layer) if args.inspect_cache else None
    capacity = runner.capacity
    if current_ids.shape[1] + args.max_new_tokens > capacity:
        raise ValueError("Prompt and generation budget exceed cache capacity")

    if args.full_prefix:
        validate_full_prefix_budget(prompt_ids.shape[1], args.max_new_tokens, runner.max_chunk)
    reference = HuggingFaceReference(args.model, args.revision) if args.compare_hf else None
    if args.inspect_cache or args.inspect_attention:
        if reference.model.config.model_type != "qwen3":
            raise ValueError("Cache/RoPE comparison currently verified only for Qwen3")

    with ExitStack() as resources:
        attention_inspector = None
        if args.inspect_attention:
            from tensorrt_debug.inspect_attention import AttentionInspector

            attention_inspector = AttentionInspector(runner, reference.model)
            resources.callback(attention_inspector.close)
        print(f"Prompt tokens: {prompt_ids.shape[1]}; cache capacity: {capacity}", flush=True)
        generate(
            args,
            runner,
            tokenizer,
            prompt_ids,
            reference,
            cache_inspector,
            attention_inspector,
        )


def generate(
    args, runner, tokenizer, prompt_ids, reference, cache_inspector, attention_inspector
):
    """Prepare -> snapshot -> execute -> snapshot -> compare -> choose next token."""
    current_ids = prompt_ids.copy()
    generated = []
    position = 0
    compared_steps = 0

    for step in range(args.max_new_tokens):
        if args.full_prefix:
            current_ids = build_full_prefix(prompt_ids, generated)
            position = 0
            runner.reset_caches()
            if reference is not None:
                reference.reset_cache()

        phase = "full-prefix" if args.full_prefix else "prefill" if step == 0 else "decode"
        length = current_ids.shape[1]
        feeds = {
            "input_ids": current_ids,
            "position_ids": np.arange(position, position + length, dtype=np.int64)[None, :],
            "write_indices": np.asarray([position], dtype=np.int64),
            "nonpad_kv_seqlen": np.asarray([position + length], dtype=np.int64),
        }
        runner.prepare_inputs(feeds)

        inspect_this_step = cache_inspector is not None and step < 2
        if inspect_this_step:
            before = cache_inspector.snapshot(show_layout=step == 0)

        last_logits = runner.execute()

        if inspect_this_step:
            after = cache_inspector.snapshot()

        if reference is not None and step < args.compare_steps:
            reference_logits = reference.forward(feeds)
            label = f"{phase} step {step} (position {position}, input length {length})"
            compare_logits(label, last_logits, reference_logits, tokenizer)
            compared_steps += 1
            if attention_inspector is not None:
                attention_inspector.compare(feeds, position)
            if inspect_this_step:
                cache_inspector.compare(before, after, reference.cache, position, length)

        token = int(last_logits.argmax())
        generated.append(token)
        print(
            f"Step {step}: {phase}; token={token}; text={tokenizer.decode([token])!r}",
            flush=True,
        )
        position += length
        if token == tokenizer.eos_token_id:
            break
        current_ids = np.asarray([[token]], dtype=np.int64)
    print("\nPrompt:", args.prompt)
    print("Response:", tokenizer.decode(generated, skip_special_tokens=True))
    if args.compare_hf:
        print(
            f"Compared {compared_steps}/{args.compare_steps} identical-token steps "
            "against FP32 HF. BF16 engine outputs need not match exactly."
        )
        if compared_steps < args.compare_steps:
            print("Comparison truncated by EOS; requested decode coverage is incomplete.")
    else:
        print("Real inference completed; numerical reference parity has not been checked.")


def main() -> None:
    args = parse_args()
    with TensorRTRunner(args.engine, args.sdk) as runner:
        run(args, runner)


if __name__ == "__main__":
    main()
