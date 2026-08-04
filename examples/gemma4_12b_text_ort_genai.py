#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

r"""Gemma-4-12B **text-only** generation with onnxruntime-genai.

``google/gemma-4-12B`` is an encoder-free **unified multimodal** checkpoint
(HuggingFace ``model_type == "gemma4_unified"``).  By default ``mobius.build``
auto-detects it and produces a 4-model multimodal package (decoder + embedding
+ vision_encoder + audio_encoder) whose decoder uses the **bidirectional
vision-block** attention overlay — a static graph choice that forces the
float-bias ``Attention`` path and forgoes ``GroupQueryAttention`` even for
text-only prompts.

This example instead exports the **text backbone** as a standalone
decoder-only LLM via ``auto_export(..., text_only=True)``.  ``text_only``:

- remaps the resolved ``model_type`` ``gemma4_unified`` -> ``gemma4_unified_text``
  (registry sibling -> :class:`~mobius.models.gemma4.Gemma4CausalLMModel`), and
- strips the vision/audio config fields (``image_token_id``,
  ``use_bidirectional_attention``, ``audio``, ...),

producing a pure-causal decoder.  Built for a GQA-capable execution provider
(``--ep cuda``), the decoder emits ``GroupQueryAttention`` (fused RoPE +
attention + KV cache), and the resulting ORT-GenAI package is a single
``model.onnx`` with a decoder-only ``genai_config.json`` (no vision/audio
sections).

Compared to the full multimodal package this is a drop-in **text** LLM: smaller
and faster to decode.  Note it keeps the same mixed per-layer KV shapes as the
multimodal decoder (sliding ``num_key_value_heads``/``head_dim`` vs global
``num_global_key_value_heads``/``global_head_dim``), so loading it still
requires a **per-layer-KV-aware** onnxruntime-genai build — a stock build
allocates KV uniformly and mis-shapes the global-attention layers (see the
multimodal example's caveat 1).

``google/gemma-4-12B`` is a **base** (non-instruction-tuned) checkpoint, so
completion-style leads work better than instructions (e.g. "The capital of
Japan is" -> " Tokyo.").

Requirements::

    pip install mobius-onnx[ort-genai] transformers
    # plus a per-layer-KV-aware onnxruntime-genai build

Usage::

    # Build + export the text-only GQA package (downloads ~24GB of weights):
    python examples/gemma4_12b_text_ort_genai.py \
        --save-to out/gemma4_12b_text/ --ep cuda --dtype f16

    # Reuse the exported dir for more prompts (no rebuild):
    python examples/gemma4_12b_text_ort_genai.py \
        --model-dir out/gemma4_12b_text/ --ep cuda \
        --prompt "The capital of Japan is"

The equivalent raw-ONNX (no genai_config) build is::

    mobius build --model google/gemma-4-12B --text-only --ep cuda --dtype f16 \
        out/gemma4_12b_text_onnx/
"""

from __future__ import annotations

import argparse
import sys

import onnxruntime_genai as og

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "google/gemma-4-12B"
# gemma uses BOS id 2.  genai's tokenizer defaults to add_special_tokens=false
# and this base checkpoint has no chat template, so BOS must be added manually.
BOS_TOKEN_ID = 2
MAX_NEW_TOKENS = 40


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def build_and_export(model_id: str, output_dir: str, dtype: str, ep: str) -> None:
    """Build the text-only GQA decoder and write ORT GenAI artifacts.

    Produces a single ``model.onnx`` plus a decoder-only ``genai_config.json``
    and tokenizer files, via the real
    ``mobius.integrations.ort_genai.auto_export`` path with ``text_only=True``.

    Args:
        model_id: HuggingFace model ID (``google/gemma-4-12B``).
        output_dir: Directory to write all outputs.
        dtype: Model dtype (``"f16"`` recommended for CUDA / GQA).
        ep: Execution provider. Drives both the build-time fusion (``cuda``
            enables GroupQueryAttention) and the genai ``session_options`` EP.
    """
    from mobius.integrations.ort_genai.auto_export import auto_export

    print(
        f"Building text-only package for {model_id!r} "
        f"(dtype={dtype}, ep={ep}) — this downloads ~24GB of weights ..."
    )
    manifest = auto_export(model_id, output_dir, dtype=dtype, ep=ep, text_only=True)
    print(f"Export complete -> {output_dir}")
    for name, path in sorted(manifest.items()):
        print(f"  {name}: {path}")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _load_model(model_dir: str, ep: str) -> tuple[og.Model, og.Tokenizer]:
    config = og.Config(model_dir)
    config.clear_providers()
    if ep != "cpu":
        config.append_provider(ep)
    model = og.Model(config)
    return model, og.Tokenizer(model)


def generate_text(model_dir: str, prompt: str, ep: str, max_new: int) -> str:
    """Greedy text generation through the native genai decoder-only path."""
    model, tokenizer = _load_model(model_dir, ep)
    # Prepend BOS manually: base checkpoint, no chat template (see BOS_TOKEN_ID).
    input_ids = [BOS_TOKEN_ID, *tokenizer.encode(prompt)]

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(input_ids) + max_new, do_sample=False)
    generator = og.Generator(model, params)
    generator.append_tokens(input_ids)

    print(f"\nPrompt: {prompt}\n" + "-" * 40)
    stream = tokenizer.create_stream()
    tokens: list[int] = []
    for _ in range(max_new):
        if generator.is_done():
            break
        generator.generate_next_token()
        tok = int(generator.get_next_tokens()[0])
        tokens.append(tok)
        print(stream.decode(tok), end="", flush=True)
    print("\n" + "-" * 40)
    del generator
    return tokenizer.decode(tokens)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Pre-exported package dir. If omitted, the model is built first "
        "(use --save-to to keep it).",
    )
    parser.add_argument(
        "--save-to",
        default=None,
        help="Directory to build + export into (reusable across runs).",
    )
    parser.add_argument(
        "--prompt",
        default="The capital of Japan is",
        help="Completion-style lead (this is a base checkpoint).",
    )
    parser.add_argument("--ep", default="cuda", help="Execution provider (cuda/cpu/dml).")
    parser.add_argument("--dtype", default="f16", help="Model dtype (f16/f32/bf16).")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    model_dir = args.model_dir
    if model_dir is None:
        if args.save_to is None:
            parser.error("Provide --model-dir (pre-built) or --save-to (to build).")
        build_and_export(MODEL_ID, args.save_to, args.dtype, args.ep)
        model_dir = args.save_to

    text = generate_text(model_dir, args.prompt, args.ep, args.max_new_tokens)
    print(f"\nGenerated: {text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
