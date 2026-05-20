# ruff: noqa: RUF002
r"""End-to-end translation example for Hy-MT1.5-1.8B via ORT GenAI.

Demonstrates loading a mobius-built Hy-MT1.5 export with
``onnxruntime-genai`` (no PyTorch at inference time). The HF model_type
``hunyuan_v1_dense`` is mapped to ORT GenAI's generic ``decoder`` LLM
type (see ``onnxruntime-genai/src/models/model_type.h``); this script
verifies that the resulting ``genai_config.json`` loads and that
``Generator`` produces real translations end-to-end.

Two model variants are supported, selected by ``--variant``:

* ``bf16``  - full-precision export from the HuggingFace safetensors.
  Note: BF16 MatMul isn't implemented on the ORT CPU EP, so this
  variant only runs on CUDA / DML / WebGPU.
* ``Q1_0``  - 2-bit-quantized export from the AngelSlim GGUF (Tencent
  SEQ codebook; default mobius packing is ``MatMulNBits bits=4``
  inflated form for fast CPU decode).

Tokenizer caveat: the Hy-MT1.5 BPE vocab uses a custom regex
pre-tokenizer that ort-extensions does not currently round-trip
(``"Hello, world!"`` tokenizes to a single space token, ``"你好"``
tokenizes to an empty sequence). By default this example tokenizes
and detokenizes with the HuggingFace tokenizer and feeds raw token
IDs to ``og.Generator``, which still exercises the full ORT GenAI
inference path. Pass ``--use-ort-tokenizer`` to force the broken
``og.Tokenizer`` path for reproduction / debugging.

Expected output (Q1_0 on CPU EP)::

    $ python examples/hy_mt1_5.py --variant Q1_0 --out ./hy-mt-q1_0-onnx \\
          --prompt 'Translate to Spanish: The cat is sleeping on the chair.'
    El gato está durmiendo en la silla.

    $ python examples/hy_mt1_5.py --variant Q1_0 --out ./hy-mt-q1_0-onnx \\
          --prompt 'Translate to French: Knowledge is power.'
    La connaissance est puissance.

    $ python examples/hy_mt1_5.py --variant Q1_0 --out ./hy-mt-q1_0-onnx
    # default prompt: Translate the following Chinese text to English: 你好，世界！
    Hello, world!

Usage::

    # Build + run the Q1_0 quantized variant on CPU
    huggingface-cli download AngelSlim/Hy-MT1.5-1.8B-2bit-GGUF \\
        Hy-MT1.5-1.8B-2bit.gguf --local-dir ./gguf
    python examples/hy_mt1_5.py --variant Q1_0 --build \\
        --gguf ./gguf/Hy-MT1.5-1.8B-2bit.gguf --out ./hy-mt-q1_0-onnx

    # Reuse an existing build (no --build)
    python examples/hy_mt1_5.py --variant Q1_0 --out ./hy-mt-q1_0-onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HF_MODEL_ID = "tencent/Hy-MT1.5-1.8B-2bit"

_DEFAULT_PROMPT = "Translate the following Chinese text to English: 你好，世界！"  # noqa: RUF001


def _build_bf16(output_dir: Path) -> None:
    """Build the BF16 ONNX model + ORT-GenAI config in-process."""
    import mobius
    from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

    print(f"Building bf16 -> {output_dir}")
    pkg = mobius.build(HF_MODEL_ID, dtype="bf16", load_weights=True)
    pkg.save(str(output_dir))
    write_ort_genai_config(pkg, str(output_dir), hf_model_id=HF_MODEL_ID, ep="cpu")
    print("  genai_config.json + tokenizer files written")


def _build_q1_0(gguf_path: Path, output_dir: Path) -> None:
    """Build the 2-bit Q1_0 ONNX model + ORT-GenAI config in-process."""
    from mobius.integrations.gguf._builder import build_from_gguf
    from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

    print(f"Building Q1_0 -> {output_dir}")
    # ep='cpu' applies the GroupQueryAttention rewrite. With standard
    # opset 23 Attention (ep='default'), ORT GenAI's
    # past_present_share_buffer mode cannot be used; mobius's
    # write_ort_genai_config inspects the resulting graph and turns
    # share_buffer off automatically when GQA is absent.
    pkg = build_from_gguf(
        str(gguf_path), keep_quantized=True, dtype="f32", execution_provider="cpu"
    )
    pkg.save(str(output_dir))
    write_ort_genai_config(pkg, str(output_dir), hf_model_id=HF_MODEL_ID, ep="cpu")
    print("  genai_config.json + tokenizer files written")


def _ensure_built(args: argparse.Namespace) -> Path:
    out = Path(args.out).resolve()
    if args.build:
        out.mkdir(parents=True, exist_ok=True)
        if args.variant == "bf16":
            _build_bf16(out)
        elif args.variant == "Q1_0":
            if not args.gguf:
                sys.exit("--gguf is required for --variant Q1_0 --build")
            _build_q1_0(Path(args.gguf), out)
    if not (out / "genai_config.json").exists():
        sys.exit(
            f"{out}/genai_config.json missing. Re-run with --build or point --out "
            "at a directory produced by `mobius build --runtime ort-genai`."
        )
    return out


def run_translation(
    model_dir: Path,
    prompt: str,
    max_new_tokens: int,
    use_hf_tokenizer: bool,
) -> str:
    """Run a single greedy generation through ORT GenAI."""
    import onnxruntime_genai as og

    print(f"\nLoading ORT GenAI model from {model_dir}")
    model = og.Model(str(model_dir))

    # Apply the chat template from the model dir (works either way).
    chat_template_path = model_dir / "chat_template.jinja"
    if chat_template_path.exists():
        from jinja2 import Environment

        template = Environment().from_string(chat_template_path.read_text())
        full_prompt = template.render(
            messages=[{"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )
    else:
        full_prompt = prompt

    print(f"\nPrompt:\n  {prompt!r}")

    if use_hf_tokenizer:
        # Tokenize and detokenize via HuggingFace. ORT-extensions' BPE
        # tokenizer does not currently round-trip the Hy-MT1.5 vocab's
        # custom regex pre-tokenizer, so we bypass og.Tokenizer here
        # while still exercising the ORT GenAI inference path.
        import transformers

        hf_tok = transformers.AutoTokenizer.from_pretrained(HF_MODEL_ID)
        input_tokens = hf_tok.encode(full_prompt, add_special_tokens=False)

        params = og.GeneratorParams(model)
        params.set_search_options(
            max_length=len(input_tokens) + max_new_tokens, do_sample=False
        )
        generator = og.Generator(model, params)
        generator.append_tokens(input_tokens)

        print("Generating: ", end="", flush=True)
        generated_ids: list[int] = []
        while not generator.is_done() and len(generated_ids) < max_new_tokens:
            generator.generate_next_token()
            new_token = int(generator.get_next_tokens()[0])
            generated_ids.append(new_token)
            # Decode incrementally; some tokens are multi-byte and only
            # render when combined with neighbours, so we re-decode the
            # accumulated tail and print the delta.
            text_so_far = hf_tok.decode(generated_ids, skip_special_tokens=False)
            print(text_so_far, end="\r", flush=True)
        print()
        return hf_tok.decode(generated_ids, skip_special_tokens=False)

    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    input_tokens = tokenizer.encode(full_prompt)

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(input_tokens) + max_new_tokens, do_sample=False)
    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)

    print("Generating: ", end="", flush=True)
    chunks: list[str] = []
    while not generator.is_done():
        generator.generate_next_token()
        new_token = generator.get_next_tokens()[0]
        chunk = tokenizer_stream.decode(new_token)
        chunks.append(chunk)
        print(chunk, end="", flush=True)
    print()
    return "".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variant", choices=("bf16", "Q1_0"), default="bf16")
    parser.add_argument(
        "--out", default=None, help="ONNX model directory (will be created with --build)"
    )
    parser.add_argument("--build", action="store_true", help="Build the model before running")
    parser.add_argument(
        "--gguf",
        default=None,
        help="Path to the Tencent SEQ GGUF (required with --variant Q1_0 --build)",
    )
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--use-ort-tokenizer",
        action="store_true",
        help=(
            "Use ort-extensions' tokenizer (og.Tokenizer) instead of the "
            "HuggingFace one. The Hy-MT1.5 vocab's custom regex pre-tokenizer "
            "is not currently round-tripped by ort-extensions, so by default "
            "we tokenize with HF and feed raw token IDs to og.Generator. "
            "Enable this flag to exercise the full og.Tokenizer path."
        ),
    )
    args = parser.parse_args()

    if args.out is None:
        args.out = f"./hy-mt-{args.variant.lower()}-onnx"

    out = _ensure_built(args)
    text = run_translation(
        out,
        args.prompt,
        args.max_new_tokens,
        use_hf_tokenizer=not args.use_ort_tokenizer,
    )

    print("\n--- Full response ---")
    print(text)


if __name__ == "__main__":
    main()
