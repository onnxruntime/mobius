# ruff: noqa: RUF002, RUF003
r"""End-to-end parity demo for Tencent Hy-MT2-1.8B (HunYuanDenseV1).

Builds the Hy-MT2-1.8B translation model to ONNX with mobius, runs greedy
generation through ONNX Runtime, then runs the **same** greedy decode with
HuggingFace ``transformers`` and asserts the two produce *identical* token
sequences. This is the L4/L5 parity check (``tests/e2e_golden_test.py``)
turned into a runnable, self-contained translation example.

The HunYuanDenseV1 architecture is GQA (16 query / 4 KV heads, head_dim 128),
per-head QK-norm, tied input/output embeddings, and dynamic-NTK RoPE with
``alpha=1000`` — all handled by mobius's generic ``CausalLMModel`` path plus
``HunYuanV1DenseCausalLMModel`` (QK-norm + weight renames).

Usage::

    # Exact, portable parity on CPU (float32 both sides):
    python examples/hy_mt2.py

    # Custom translation prompt:
    python examples/hy_mt2.py --source "今天天气真好。" --target-lang English

    # Half precision on CUDA (bf16 MatMul is unsupported on the ORT CPU EP):
    python examples/hy_mt2.py --dtype f16 --device cuda

The default prompt mirrors the Hy-MT chat template the model was trained
with (``<｜hy_begin▁of▁sentence｜><｜hy_User｜>...<｜hy_Assistant｜>``); the
template is applied via the HF tokenizer so the formatting always matches
what the checkpoint expects.
"""

from __future__ import annotations

import argparse

import numpy as np

HF_MODEL_ID = "tencent/Hy-MT2-1.8B"
# <｜hy_place▁holder▁no▁2｜>, the turn-end / EOS used by generation_config.
EOS_TOKEN_ID = 120020

_DEFAULT_SOURCE = "黄河之水天上来"
_DEFAULT_TARGET = "English"


def _dtype_to_torch(dtype: str):
    import torch

    return {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[dtype]


def _format_prompt(tokenizer, source: str, target_lang: str) -> str:
    """Apply the Hy-MT chat template to a single translation instruction."""
    instruction = (
        f"Translate the following text into {target_lang}. Note that you "
        f"should only output the translated result without any additional "
        f"explanation:\n\n{source}"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )


def run_onnx(
    pkg, config, input_ids: np.ndarray, max_new_tokens: int, device: str
) -> list[int]:
    """Greedy-decode through ONNX Runtime and return the generated token IDs."""
    from mobius._testing.generation import OnnxGenerator
    from mobius._testing.ort_inference import OnnxModelSession

    session = OnnxModelSession(pkg["model"], device=device)
    generator = OnnxGenerator(session, config)
    out = generator.generate(
        input_ids, max_new_tokens=max_new_tokens, eos_token_id=EOS_TOKEN_ID
    )
    return out[0, input_ids.shape[1] :].tolist()


def run_hf(model, input_ids: np.ndarray, max_new_tokens: int) -> list[int]:
    """Greedy-decode through HuggingFace and return the generated token IDs."""
    import torch

    with torch.no_grad():
        out = model.generate(
            input_ids=torch.from_numpy(input_ids).to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=EOS_TOKEN_ID,
            pad_token_id=EOS_TOKEN_ID,
        )
    return out[0, input_ids.shape[1] :].tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", default=_DEFAULT_SOURCE, help="Text to translate.")
    parser.add_argument("--target-lang", default=_DEFAULT_TARGET, help="Target language name.")
    parser.add_argument("--dtype", choices=("f32", "f16", "bf16"), default="f32")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="ORT/torch device. f16/bf16 require cuda (no CPU MatMul kernels).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    if args.dtype in ("f16", "bf16") and args.device != "cuda":
        parser.error(f"--dtype {args.dtype} needs --device cuda (no ORT CPU MatMul kernel)")

    import transformers

    import mobius

    print(f"Loading tokenizer + HF reference model ({HF_MODEL_ID}, {args.dtype})...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(HF_MODEL_ID)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID, dtype=_dtype_to_torch(args.dtype)
    ).to(args.device)
    hf_model.eval()

    prompt = _format_prompt(tokenizer, args.source, args.target_lang)
    input_ids = np.array([tokenizer.encode(prompt, add_special_tokens=False)], dtype=np.int64)
    print(f"\nPrompt:\n  {prompt!r}\n  ({input_ids.shape[1]} tokens)")

    print(f"\nBuilding ONNX export ({args.dtype})...")
    pkg = mobius.build(HF_MODEL_ID, dtype=args.dtype, load_weights=True)
    config = pkg.config

    print("Running greedy generation: ONNX Runtime ...")
    onnx_ids = run_onnx(pkg, config, input_ids, args.max_new_tokens, args.device)
    print("Running greedy generation: HuggingFace ...")
    hf_ids = run_hf(hf_model, input_ids, args.max_new_tokens)

    onnx_text = tokenizer.decode(onnx_ids, skip_special_tokens=True)
    hf_text = tokenizer.decode(hf_ids, skip_special_tokens=True)

    print("\n--- Translation ---")
    print(f"  source : {args.source}")
    print(f"  ONNX   : {onnx_text}")
    print(f"  HF     : {hf_text}")

    print("\n--- Token-level parity ---")
    print(f"  ONNX tokens: {onnx_ids}")
    print(f"  HF   tokens: {hf_ids}")
    if onnx_ids == hf_ids:
        print(f"\n✅ PASS: ONNX and HuggingFace produced identical {len(onnx_ids)} tokens.")
    else:
        n = min(len(onnx_ids), len(hf_ids))
        first = next((i for i in range(n) if onnx_ids[i] != hf_ids[i]), n)
        raise SystemExit(
            f"\n❌ FAIL: token sequences diverge at index {first} "
            f"(ONNX={onnx_ids[first : first + 1]} vs HF={hf_ids[first : first + 1]})."
        )


if __name__ == "__main__":
    main()
