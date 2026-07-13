#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

r"""Gemma-4-12B (``gemma4_unified``) multimodal generation with onnxruntime-genai.

``google/gemma-4-12B`` is an **encoder-free unified multimodal** checkpoint
(HuggingFace ``model_type == "gemma4_unified"``).  ``mobius.build`` auto-detects
it and produces a **4-model package**: ``decoder`` + ``embedding`` +
``vision_encoder`` + ``audio_encoder``.  Unlike the released ``gemma4`` models
there is no SigLIP / Conformer tower — raw merged pixel patches (48x48, 6912-dim)
and raw waveform-frame features (640-dim) are projected directly into language
space by ``vision_encoder`` / ``audio_encoder``, then scattered into
``inputs_embeds`` by ``embedding`` at the image / audio placeholder positions.

This script builds the full package via the real
``mobius.integrations.ort_genai`` export path and runs **text**, **image+text**,
and **audio+text** generation through onnxruntime-genai on the same package.

Two runtime caveats are handled below:

1. **Patched onnxruntime-genai required.**  gemma-4-12B has a *mixed* KV cache:
   most layers use 8 heads x head_dim 256, but the global-attention layers use a
   single 1 head x head_dim 512 KV.  A stock genai build assumes uniform KV
   shapes and fails to load the decoder.  Use a build that reads per-layer
   ``num_heads`` / ``head_dim`` from each ``past_key_values.*`` input shape.

2. **HuggingFace preprocessing for image / audio.**  genai's built-in
   ``Gemma4ImageTransform`` targets the SigLIP ``gemma4`` patch contract
   (16px, 768-dim), which does **not** match this encoder-free unified model
   (48px merged patches, 6912-dim).  Until a genai-native unified transform
   exists, this example preprocesses image / audio with the HuggingFace
   ``AutoProcessor`` and feeds the resulting tensors to genai via
   ``Generator.set_inputs(NamedTensors)`` (which bypasses genai's own
   transform).  Text generation uses the native genai path.

3. **Structural-token suppression.**  HF's ``generation_config`` suppresses the
   ``<end_of_image>`` / ``<end_of_audio>`` tokens (``suppress_tokens``).  genai
   has no native equivalent, so without it this base checkpoint degenerates into
   repeating ``<image|>`` after an image instead of describing it.  The decode
   loop masks those token ids to ``-inf`` before sampling (see SUPPRESS_TOKEN_IDS
   and ``_decode_loop``).

``google/gemma-4-12B`` is a **base** (non-instruction-tuned) checkpoint, so
completion-style leads work far better than instructions (the defaults use
"This image shows" / "The audio says", not "Describe ...").  Verified outputs on
GPU (f16, greedy, matching HuggingFace ``model.generate``):

- text  "The capital of Japan is"  -> " Tokyo."
- image (Sydney Chinatown photo)   -> " the Chinese Arch in the Chinatown of
  Sydney, Australia."
- audio (LibriSpeech clip)          -> " He hoped there would be stew for
  dinner, turnips and carrots and bruised potatoes ..."

Numerical correctness of the image / audio pipeline is also verified by the
integration tests (``tests/integration_test.py::
test_gemma4_unified_12b_multimodal_prefill``, vision cosine 1.0 vs HuggingFace).

Optional ``--quantize Q4_K_M`` INT4-quantizes the decoder with Olive (~23GB ->
~6.8GB, 3.4x smaller).  Spot-checked quality on this base model: coherent text
/ image / audio generation, ~0.986 last-token logit cosine and ~75% greedy
top-1 agreement vs f16 on short factual prompts (the base model itself emits
some off-distribution tokens, so a few disagreements are not quantization
artifacts).

Requirements::

    pip install mobius-ai[ort-genai] transformers pillow librosa
    # plus a per-layer-KV-aware onnxruntime-genai build (see caveat 1)

Usage::

    # Build the package once and reuse it across modes:
    python examples/gemma4_unified_ort_genai.py --mode text --save-to out/gemma4_12b/

    # Image + text (reuse the built dir):
    python examples/gemma4_unified_ort_genai.py --mode image \
        --model-dir out/gemma4_12b/ --image path/to/photo.jpg

    # Audio + text:
    python examples/gemma4_unified_ort_genai.py --mode audio \
        --model-dir out/gemma4_12b/ --audio path/to/clip.flac

    # INT4-quantize the decoder with Olive (Q4_K_M, ~3.4x smaller) and run it:
    python examples/gemma4_unified_ort_genai.py --mode image \
        --model-dir out/gemma4_12b/ --image path/to/photo.jpg \
        --quantize Q4_K_M --quantized-out out/gemma4_12b-Q4_K_M/
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import onnxruntime_genai as og

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "google/gemma-4-12B"
# gemma uses BOS id 2.  genai's tokenizer defaults to add_special_tokens=false
# and this base checkpoint has no chat template, so BOS must be added manually.
BOS_TOKEN_ID = 2
IMAGE_TOKEN_ID = 258880
AUDIO_TOKEN_ID = 258881
# Structural multimodal tokens that the base model tends to emit verbatim during
# generation (``<end_of_image>`` / ``<end_of_audio>``).  HF's generation_config
# lists them in ``suppress_tokens``; genai has no native suppression, so the
# decode loop forces their logits to -inf (see _decode_loop).  Without this the
# base checkpoint degenerates into repeating ``<image|>`` instead of captioning.
SUPPRESS_TOKEN_IDS = (258882, 258883)
MAX_NEW_TOKENS = 40


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def build_and_export(model_id: str, output_dir: str, dtype: str, ep: str) -> None:
    """Build the full unified multimodal package and write ORT GenAI artifacts.

    Produces ``decoder`` + ``embedding`` + ``vision_encoder`` + ``audio_encoder``
    ONNX models plus ``genai_config.json``, tokenizer files, ``image_processor``
    and audio feature-extraction configs, via the real
    ``mobius.integrations.ort_genai.auto_export`` path.

    Args:
        model_id: HuggingFace model ID (``google/gemma-4-12B``).
        output_dir: Directory to write all outputs.
        dtype: Model dtype (``"f16"`` recommended for CUDA).
        ep: Execution provider for ``genai_config.json`` session options.
    """
    from mobius.integrations.ort_genai import auto_export

    print(
        f"Building unified multimodal package for {model_id!r} "
        f"(dtype={dtype}, ep={ep}) — this downloads ~25GB of weights ..."
    )
    manifest = auto_export(model_id, output_dir, dtype=dtype, ep=ep)
    print(f"Export complete -> {output_dir}")
    for name, path in sorted(manifest.items()):
        print(f"  {name}: {path}")


# ---------------------------------------------------------------------------
# Olive INT4 quantization (decoder only)
# ---------------------------------------------------------------------------


def quantize_decoder(
    src_dir: str,
    dst_dir: str,
    *,
    precision: str = "Q4_K_M",
    block_size: int = 32,
) -> None:
    """INT4-quantize the decoder sub-model with Olive (k-quant or NF4).

    Only the decoder is quantized — it holds ~23GB of the package's weights
    (>95%).  The embedding / vision_encoder / audio_encoder sub-models, the
    tokenizer, ``genai_config.json``, and the processor configs are copied
    over unchanged, so the result is a drop-in ORT GenAI package.

    Args:
        src_dir: full-precision package (output of :func:`build_and_export`).
        dst_dir: destination directory for the quantized package.
        precision: ``"Q4_K_M"`` (k-quant; install ``cupy-cuda12x`` for the
            19-51x GPU speedup) or ``"NF4"`` (4-bit NormalFloat, native C++).
        block_size: k-quant block size; ignored for NF4.

    Requires ``olive-ai`` (``pip install olive-ai``).
    """
    import shutil

    from olive.workflows import run as olive_run

    if precision == "Q4_K_M":
        pass_cfg = {"type": "OnnxKQuantQuantization", "bits": 4, "block_size": block_size}
    elif precision == "NF4":
        pass_cfg = {"type": "OnnxBnb4Quantization", "precision": "nf4"}
    else:
        raise ValueError(f"Unsupported precision: {precision!r}")

    decoder_dst = os.path.join(dst_dir, "decoder")
    os.makedirs(decoder_dst, exist_ok=True)
    config = {
        "input_model": {
            "type": "OnnxModel",
            "model_path": os.path.join(src_dir, "decoder", "model.onnx"),
        },
        "passes": {precision.lower(): pass_cfg},
        "output_dir": decoder_dst,
    }
    print(f"Quantizing decoder ({precision}) -> {decoder_dst} ...")
    olive_run(config)

    # Olive writes bookkeeping files that ORT GenAI must not see in the
    # decoder/ folder; keep only the ONNX model + its external data.
    for f in os.listdir(decoder_dst):
        if not (f == "model.onnx" or f.startswith("model.onnx.data")):
            path = os.path.join(decoder_dst, f)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)

    _ensure_logits_output(os.path.join(decoder_dst, "model.onnx"))

    # Copy the non-decoder sub-models + config + tokenizer unchanged.
    for child in os.listdir(src_dir):
        path = os.path.join(src_dir, child)
        target = os.path.join(dst_dir, child)
        if child == "decoder" or os.path.exists(target):
            continue
        shutil.copytree(path, target) if os.path.isdir(path) else shutil.copy2(path, target)
    print(f"Quantized package ready -> {dst_dir}")


def _ensure_logits_output(decoder_path: str) -> None:
    """Rename a quantized decoder's ``logits_Q4`` output back to ``logits``.

    Some Olive versions rename the decoder's ``logits`` output to ``logits_Q4``
    during k-quant.  ORT GenAI maps outputs by the names in the (copied)
    ``genai_config.json`` (which says ``logits``), so we rename the graph
    output back when needed to keep the quantized decoder a drop-in.
    """
    import onnx_ir as ir

    model = ir.load(decoder_path)
    renamed = False
    for value in model.graph.outputs:
        if value.name == "logits_Q4":
            value.name = "logits"
            renamed = True
    if renamed:
        ir.save(model, decoder_path, external_data="model.onnx.data")
        print("  Renamed decoder output logits_Q4 -> logits")


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def _load_model(model_dir: str, ep: str) -> tuple[og.Model, og.Tokenizer]:
    config = og.Config(model_dir)
    config.clear_providers()
    if ep != "cpu":
        config.append_provider(ep)
    model = og.Model(config)
    return model, og.Tokenizer(model)


def _decode_loop(
    model: og.Model, generator: og.Generator, tokenizer: og.Tokenizer, max_new: int
) -> str:
    stream = tokenizer.create_stream()
    tokens: list[int] = []
    for _ in range(max_new):
        if generator.is_done():
            break
        # Suppress the structural multimodal tokens before sampling, mirroring
        # HF generation_config's ``suppress_tokens`` (genai has no native
        # equivalent).  get_logits -> mask -> set_logits -> sample.
        logits = generator.get_logits()
        logits[..., list(SUPPRESS_TOKEN_IDS)] = float("-inf")
        generator.set_logits(logits)
        generator.generate_next_token()
        tok = int(generator.get_next_tokens()[0])
        tokens.append(tok)
        print(stream.decode(tok), end="", flush=True)
    print()
    return tokenizer.decode(tokens)


def generate_text(model_dir: str, prompt: str, ep: str, max_new: int) -> str:
    """Greedy text generation through the native genai path (BOS prepended)."""
    model, tokenizer = _load_model(model_dir, ep)
    # Prepend BOS manually: base checkpoint, no chat template (see BOS_TOKEN_ID).
    input_ids = [BOS_TOKEN_ID, *tokenizer.encode(prompt)]

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(input_ids) + max_new, do_sample=False)
    generator = og.Generator(model, params)
    generator.append_tokens(input_ids)

    print(f"\nPrompt: {prompt}\n" + "-" * 40)
    text = _decode_loop(model, generator, tokenizer, max_new)
    print("-" * 40)
    del generator
    return text


def generate_image(
    model_dir: str, model_id: str, image_path: str, prompt: str, ep: str, max_new: int
) -> str:
    """Image+text generation: HF unified processor -> genai ``set_inputs``.

    genai's built-in image transform targets the SigLIP ``gemma4`` contract, so
    we preprocess with the HuggingFace processor (48px merged patches, 6912-dim)
    and inject the tensors directly.  ``set_inputs`` bypasses genai's transform;
    genai then runs ``vision_encoder -> embedding -> decoder``.
    """
    from PIL import Image
    from transformers import AutoProcessor

    model, tokenizer = _load_model(model_dir, ep)
    processor = AutoProcessor.from_pretrained(model_id)
    image = Image.open(image_path).convert("RGB")

    # The HF processor inserts IMAGE_TOKEN_ID placeholders and (for gemma) BOS.
    proc = processor(
        text=[f"{processor.image_token}{prompt}"], images=[image], return_tensors="pt"
    )
    input_ids = proc["input_ids"].numpy().astype(np.int32)
    n_image_tokens = int((input_ids == IMAGE_TOKEN_ID).sum())

    nt = og.NamedTensors()
    nt["input_ids"] = input_ids
    # Graph input names: pixel_values, pixel_position_ids (HF names them
    # pixel_values / image_position_ids).
    nt["pixel_values"] = proc["pixel_values"].numpy().astype(np.float16)
    nt["pixel_position_ids"] = proc["image_position_ids"].numpy().astype(np.int64)
    nt["num_image_tokens"] = np.array([n_image_tokens], dtype=np.int64)

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=input_ids.shape[1] + max_new, do_sample=False)
    generator = og.Generator(model, params)
    generator.set_inputs(nt)

    print(f"\nImage: {image_path}\nPrompt: {prompt}\n" + "-" * 40)
    text = _decode_loop(model, generator, tokenizer, max_new)
    print("-" * 40)
    del generator
    return text


def generate_audio(
    model_dir: str, model_id: str, audio_path: str, prompt: str, ep: str, max_new: int
) -> str:
    """Audio+text generation: HF unified processor -> genai ``set_inputs``.

    Mirrors :func:`generate_image` for the audio branch.  genai derives the
    audio-token count from the summed ``audio_sizes`` input, then runs
    ``audio_encoder -> embedding -> decoder``.
    """
    import librosa
    from transformers import AutoProcessor

    model, tokenizer = _load_model(model_dir, ep)
    processor = AutoProcessor.from_pretrained(model_id)
    waveform, _ = librosa.load(audio_path, sr=16000)

    proc = processor(
        text=[f"{processor.audio_token}{prompt}"],
        audio=[waveform],
        return_tensors="pt",
    )
    input_ids = proc["input_ids"].numpy().astype(np.int32)
    n_audio_tokens = int((input_ids == AUDIO_TOKEN_ID).sum())

    nt = og.NamedTensors()
    nt["input_ids"] = input_ids
    nt["input_features"] = proc["input_features"].numpy().astype(np.float16)
    nt["input_features_mask"] = proc["input_features_mask"].numpy().astype(bool)
    # genai sums audio_sizes to get the audio-token count for the speech branch.
    nt["audio_sizes"] = np.array([n_audio_tokens], dtype=np.int64)

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=input_ids.shape[1] + max_new, do_sample=False)
    generator = og.Generator(model, params)
    generator.set_inputs(nt)

    print(f"\nAudio: {audio_path}\nPrompt: {prompt}\n" + "-" * 40)
    text = _decode_loop(model, generator, tokenizer, max_new)
    print("-" * 40)
    del generator
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--mode",
        default="text",
        choices=["text", "image", "audio"],
        help="Which modality to demonstrate.",
    )
    parser.add_argument("--model-id", default=MODEL_ID, help="HuggingFace model ID.")
    parser.add_argument("--prompt", default=None, help="Generation prompt.")
    parser.add_argument("--image", default=None, help="Image path (image mode).")
    parser.add_argument("--audio", default=None, help="Audio path (audio mode).")
    parser.add_argument(
        "--max-new-tokens", type=int, default=MAX_NEW_TOKENS, help="Max new tokens."
    )
    parser.add_argument(
        "--dtype", default="f16", choices=["f32", "f16", "bf16"], help="Model dtype."
    )
    parser.add_argument("--ep", default="cuda", help="Execution provider.")
    parser.add_argument(
        "--model-dir", default=None, help="Reuse a pre-built export directory."
    )
    parser.add_argument(
        "--save-to", default=None, help="Build + save to this directory (skip cleanup)."
    )
    parser.add_argument(
        "--quantize",
        default=None,
        choices=["Q4_K_M", "NF4"],
        help="INT4-quantize the decoder with Olive and run against the result.",
    )
    parser.add_argument(
        "--quantized-out",
        default=None,
        help="Output dir for --quantize (default: <model-dir>-<precision>).",
    )
    args = parser.parse_args()

    # Default prompts per mode. This is a *base* (non-instruction-tuned)
    # checkpoint, so completion-style leads work far better than instructions.
    prompt = (
        args.prompt
        or {
            "text": "The capital of France is",
            "image": "This image shows",
            "audio": "The audio says",
        }[args.mode]
    )

    if args.mode == "image" and not args.image:
        parser.error("--image is required for --mode image")
    if args.mode == "audio" and not args.audio:
        parser.error("--audio is required for --mode audio")

    # Determine the export directory.
    tmp_dir: str | None = None
    if args.model_dir is not None:
        model_dir = args.model_dir
    else:
        model_dir = args.save_to
        if model_dir is None:
            tmp_dir = tempfile.mkdtemp(prefix="gemma4_12b_mm_")
            model_dir = tmp_dir
        build_and_export(args.model_id, model_dir, args.dtype, args.ep)

    # Optionally INT4-quantize the decoder and run against the quantized package.
    if args.quantize:
        quant_dir = args.quantized_out or f"{model_dir.rstrip('/')}-{args.quantize}"
        quantize_decoder(model_dir, quant_dir, precision=args.quantize)
        model_dir = quant_dir

    try:
        if args.mode == "text":
            out = generate_text(model_dir, prompt, args.ep, args.max_new_tokens)
        elif args.mode == "image":
            out = generate_image(
                model_dir,
                args.model_id,
                args.image,
                prompt,
                args.ep,
                args.max_new_tokens,
            )
        else:
            out = generate_audio(
                model_dir,
                args.model_id,
                args.audio,
                prompt,
                args.ep,
                args.max_new_tokens,
            )
        print(f"\nORT GenAI output:\n{out}")
    finally:
        if tmp_dir is not None and args.save_to is None:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
