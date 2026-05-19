r"""End-to-end translation example for Hy-MT1.5-1.8B via ORT GenAI.

Demonstrates loading a mobius-built Hy-MT1.5 export with
``onnxruntime-genai`` (no PyTorch at inference time). The HF model_type
``hunyuan_v1_dense`` is mapped to ORT GenAI's generic ``decoder`` LLM
type (see ``onnxruntime-genai/src/models/model_type.h``); this script
verifies that the resulting ``genai_config.json`` loads and that
``Generator.generate_next_token()`` produces output end-to-end.

Two model variants are supported, selected by ``--variant``:

* ``bf16``  - full-precision export from the HuggingFace safetensors.
  Note: BF16 MatMul isn't implemented on the ORT CPU EP, so this
  variant only runs on CUDA / DML / WebGPU.
* ``Q1_0``  - 2-bit-quantized export from the AngelSlim GGUF (Tencent
  SEQ codebook; default mobius packing is ``MatMulNBits bits=4``
  inflated form for fast CPU decode).

Known caveat: the upstream Hy-MT1.5 BPE vocab uses placeholder special
tokens and Chinese characters that ort-extensions' tokenizer does not
fully round-trip today — for example ``"你好"`` tokenizes to an empty
sequence. The chat template path renders correctly, but tokenization
of CJK content in the message body is lossy. This is independent of
the ``model_type=decoder`` fix and tracked downstream.

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
    _patch_genai_config(output_dir)
    print("  genai_config.json + tokenizer files written")


def _build_q1_0(gguf_path: Path, output_dir: Path) -> None:
    """Build the 2-bit Q1_0 ONNX model + ORT-GenAI config in-process."""
    from mobius.integrations.gguf._builder import build_from_gguf
    from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

    print(f"Building Q1_0 -> {output_dir}")
    pkg = build_from_gguf(str(gguf_path), keep_quantized=True, dtype="f32")
    pkg.save(str(output_dir))
    write_ort_genai_config(pkg, str(output_dir), hf_model_id=HF_MODEL_ID, ep="cpu")
    _patch_genai_config(output_dir)
    print("  genai_config.json + tokenizer files written")


def _patch_genai_config(output_dir: Path) -> None:
    """Disable ``past_present_share_buffer`` for the dynamic-cache export.

    mobius emits a dynamic-shape KV cache (no pre-allocated buffer);
    ORT GenAI's default of ``past_present_share_buffer = True`` would
    feed the attention op a mask whose ``total_sequence_length`` doesn't
    match the dynamic past_key/past_value shapes, producing
    ``inconsistent total_sequence_length`` at runtime.
    """
    import json

    cfg_path = output_dir / "genai_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg.setdefault("search", {})["past_present_share_buffer"] = False
    cfg_path.write_text(json.dumps(cfg, indent=4))


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


def run_translation(model_dir: Path, prompt: str, max_new_tokens: int) -> str:
    """Run a single greedy generation through ORT GenAI."""
    import onnxruntime_genai as og

    print(f"\nLoading ORT GenAI model from {model_dir}")
    model = og.Model(str(model_dir))
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()

    # Apply the chat template from the model dir if present.
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
    input_tokens = tokenizer.encode(full_prompt)

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(input_tokens) + max_new_tokens, do_sample=False)
    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)

    print("Generating: ", end="", flush=True)
    response_chunks = []
    while not generator.is_done():
        generator.generate_next_token()
        new_token = generator.get_next_tokens()[0]
        chunk = tokenizer_stream.decode(new_token)
        response_chunks.append(chunk)
        print(chunk, end="", flush=True)
    print()

    text = "".join(response_chunks)
    return text


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
    args = parser.parse_args()

    if args.out is None:
        args.out = f"./hy-mt-{args.variant.lower()}-onnx"

    out = _ensure_built(args)
    text = run_translation(out, args.prompt, args.max_new_tokens)

    print("\n--- Full response ---")
    print(text)


if __name__ == "__main__":
    main()
