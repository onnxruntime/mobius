#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fun-ASR-Nano speech recognition with ONNX models.

Builds three ONNX models from ``mobius`` (audio encoder,
embedding, decoder) and runs the full ASR pipeline:

    audio → fbank → LFR → audio encoder → embedding fusion → decoder → text

The audio frontend (fbank + LFR + CMVN) runs in Python, *not* inside
the ONNX graph.  This differs from Qwen3-ASR which uses
WhisperFeatureExtractor.

Supports real-time microphone input and audio file input.

Prerequisites::

    pip install mobius-ai[transformers] sounddevice torchaudio pyyaml

Usage::

    # Record from microphone (press Enter to stop)
    python examples/fun_asr.py

    # Transcribe an audio file
    python examples/fun_asr.py --audio speech.wav

    # Continuous mic mode (Ctrl+C to exit)
    python examples/fun_asr.py --continuous

    # Force a specific language
    python examples/fun_asr.py --language zh          # Chinese
    python examples/fun_asr.py --language en           # English
    python examples/fun_asr.py --language ja           # Japanese

    # Use a different model
    python examples/fun_asr.py --model FunAudioLLM/Fun-ASR-Nano-2512

    # GPU inference with half precision
    python examples/fun_asr.py --device cuda --dtype f16

    # Disable streaming output
    python examples/fun_asr.py --no-stream

    # Save ONNX models without running inference
    python examples/fun_asr.py --save-to output/fun-asr/
"""

from __future__ import annotations

import argparse
import sys
import threading

import ml_dtypes
import numpy as np

from mobius import ArchitectureConfig, build_from_module
from mobius._configs import AudioConfig
from mobius._testing.ort_inference import OnnxModelSession

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "FunAudioLLM/Fun-ASR-Nano-2512"
SAMPLE_RATE = 16000
MAX_RECORD_SECONDS = 60
MAX_NEW_TOKENS = 4096

# LFR (Low Frame Rate) parameters from config.yaml
LFR_M = 7  # Stack 7 consecutive frames
LFR_N = 6  # Subsample by 6
N_MELS = 80  # Mel filter bank bins

# Language mapping for Fun-ASR-Nano-2512 (zh, en, ja).
# Values are Chinese language names used in the prompt (fullwidth colon is required).
LANGUAGE_MAP: dict[str, str] = {
    "auto": "",
    "zh": "中文",
    "chinese": "中文",
    "中文": "中文",
    "en": "英文",
    "english": "英文",
    "英文": "英文",
    "ja": "日文",
    "japanese": "日文",
    "日文": "日文",
}


# ---------------------------------------------------------------------------
# Audio frontend: fbank → LFR → CMVN
# ---------------------------------------------------------------------------


def compute_fbank(
    audio: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    frame_length: int = 25,
    frame_shift: int = 10,
) -> np.ndarray:
    """Compute 80-dim fbank features using Kaldi-compatible extraction.

    Returns array of shape ``(num_frames, n_mels)``.
    """
    import torch
    import torchaudio

    waveform = torch.from_numpy(audio).float()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform.unsqueeze(0),
        num_mel_bins=n_mels,
        frame_length=frame_length,
        frame_shift=frame_shift,
        sample_frequency=sample_rate,
        window_type="hamming",
    )
    return fbank.numpy()  # (num_frames, 80)


def apply_lfr(fbank: np.ndarray, lfr_m: int = LFR_M, lfr_n: int = LFR_N) -> np.ndarray:
    """Apply Low Frame Rate stacking and subsampling.

    Stacks ``lfr_m`` consecutive frames and subsamples every ``lfr_n`` frames,
    producing features of dimension ``lfr_m * n_mels`` (typically 7*80 = 560).

    Returns array of shape ``(T_out, lfr_m * n_mels)``.
    """
    num_frames = fbank.shape[0]
    # Pad to multiple of lfr_n
    pad_len = (lfr_n - (num_frames % lfr_n)) % lfr_n
    if pad_len > 0:
        fbank = np.pad(fbank, ((0, pad_len), (0, 0)), mode="edge")
    t_padded = fbank.shape[0]

    lfr_frames = []
    for i in range(0, t_padded, lfr_n):
        end = min(i + lfr_m, t_padded)
        chunk = fbank[i:end]
        if chunk.shape[0] < lfr_m:
            chunk = np.pad(chunk, ((0, lfr_m - chunk.shape[0]), (0, 0)), mode="edge")
        lfr_frames.append(chunk.flatten())
    return np.array(lfr_frames)  # (T_out, 560)


def apply_cmvn(features: np.ndarray) -> np.ndarray:
    """Apply CMVN (Cepstral Mean and Variance Normalization).

    The Fun-ASR-Nano config has ``cmvn_file: null``, so this is currently
    an identity transform.

    TODO: If a cmvn_file is provided in the model config, load the mean/var
    statistics and apply (features - mean) / sqrt(var + eps).
    """
    return features


def preprocess_audio(
    audio: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Full audio frontend: fbank → LFR → CMVN.

    Returns array of shape ``(1, T_out, 560)`` ready for the audio encoder.
    """
    fbank = compute_fbank(audio, sample_rate=sample_rate)
    lfr_features = apply_lfr(fbank)
    features = apply_cmvn(lfr_features)
    return features[np.newaxis, :, :]  # (1, T_out, 560)


# ---------------------------------------------------------------------------
# Microphone recording
# ---------------------------------------------------------------------------


def record_until_enter(
    sample_rate: int = SAMPLE_RATE,
    max_seconds: int = MAX_RECORD_SECONDS,
) -> np.ndarray:
    """Record audio from the default mic until Enter is pressed."""
    import sounddevice as sd

    chunks: list[np.ndarray] = []
    stop_event = threading.Event()

    def callback(indata, frames, time, status):
        if status:
            print(f"  [mic] {status}", file=sys.stderr)
        chunks.append(indata.copy())

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=callback,
        blocksize=int(sample_rate * 0.1),
    )

    print("🎤 Recording... Press Enter to stop.")
    stream.start()

    input_thread = threading.Thread(target=lambda: (input(), stop_event.set()))
    input_thread.daemon = True
    input_thread.start()
    input_thread.join(timeout=max_seconds)
    stop_event.set()

    stream.stop()
    stream.close()

    if not chunks:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(chunks, axis=0).flatten()
    duration = len(audio) / sample_rate
    print(f"  Recorded {duration:.1f}s of audio.")
    return audio


def load_audio_file(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file and resample to target sample rate."""
    import torchaudio

    waveform, sr = torchaudio.load(path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    # Mono, float32
    return waveform.mean(dim=0).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Model config construction
# ---------------------------------------------------------------------------


def build_fun_asr_config(model_id: str, dtype: str = "f32") -> ArchitectureConfig:
    """Build an ArchitectureConfig for Fun-ASR from the HF repo files.

    Fun-ASR uses a config.yaml (not a standard HF config.json with model_type),
    so we construct the config manually from:
    - config.yaml: audio encoder and adaptor hyperparameters
    - Qwen3-0.6B/config.json: LLM backbone hyperparameters
    """
    import json

    import yaml
    from huggingface_hub import hf_hub_download

    # Load config.yaml
    yaml_path = hf_hub_download(model_id, "config.yaml")
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    # Load LLM config (Qwen3-0.6B subfolder)
    llm_subfolder = cfg.get("llm_conf", {}).get("init_param_path", "Qwen3-0.6B")
    llm_config_path = hf_hub_download(model_id, f"{llm_subfolder}/config.json")
    with open(llm_config_path) as f:
        llm_cfg = json.load(f)

    enc = cfg.get("audio_encoder_conf", {})
    adaptor = cfg.get("audio_adaptor_conf", {})
    frontend = cfg.get("frontend_conf", {})

    # input_size = lfr_m * n_mels (7 * 80 = 560)
    lfr_m = frontend.get("lfr_m", LFR_M)
    n_mels = frontend.get("n_mels", N_MELS)
    input_size = lfr_m * n_mels

    from onnx_ir import DataType

    dtype_map = {"f32": DataType.FLOAT, "f16": DataType.FLOAT16, "bf16": DataType.BFLOAT16}

    return ArchitectureConfig(
        model_type="fun_asr",
        vocab_size=llm_cfg["vocab_size"],
        hidden_size=llm_cfg["hidden_size"],
        num_hidden_layers=llm_cfg["num_hidden_layers"],
        num_attention_heads=llm_cfg["num_attention_heads"],
        num_key_value_heads=llm_cfg.get("num_key_value_heads", llm_cfg["num_attention_heads"]),
        intermediate_size=llm_cfg.get("intermediate_size", llm_cfg["hidden_size"] * 4),
        hidden_act=llm_cfg.get("hidden_act", "silu"),
        head_dim=llm_cfg.get(
            "head_dim", llm_cfg["hidden_size"] // llm_cfg["num_attention_heads"]
        ),
        rms_norm_eps=llm_cfg.get("rms_norm_eps", 1e-6),
        rope_theta=llm_cfg.get("rope_theta", 1000000.0),
        rope_type="default",
        max_position_embeddings=llm_cfg.get("max_position_embeddings", 40960),
        attn_qk_norm=True,
        dtype=dtype_map.get(dtype, DataType.FLOAT),
        audio=AudioConfig(
            input_size=input_size,
            attention_dim=enc.get("output_size", 512),
            attention_heads=enc.get("attention_heads", 4),
            num_blocks=enc.get("num_blocks", 50),
            linear_units=enc.get("linear_units", 2048),
            kernel_size=enc.get("kernel_size", 11),
            tp_num_blocks=enc.get("tp_blocks", 20),
            output_dim=enc.get("output_size", 512),
            audio_token_id=0,  # Fun-ASR uses token_id=0 as audio placeholder
            adaptor_proj_dim=adaptor.get("ffn_dim", 2048),
            adaptor_num_blocks=adaptor.get("n_layer", 2),
            adaptor_ffn_dim=256,  # FFN hidden dim inside adaptor blocks (from weights)
            adaptor_num_heads=8,  # Reference defaults to 8 for adaptor attention
        ),
    )


# ---------------------------------------------------------------------------
# ASR inference pipeline
# ---------------------------------------------------------------------------


def transcribe(
    sessions: dict[str, OnnxModelSession],
    tokenizer,
    audio: np.ndarray,
    config: ArchitectureConfig,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
    language: str = "",
    stream: bool = True,
    model_dtype: np.dtype = np.float32,
) -> str:
    """Full ASR pipeline: audio → text.

    Runs three ONNX models in sequence:
    1. Audio encoder: LFR fbank features → audio features
    2. Embedding: fuse text tokens with audio features
    3. Decoder: autoregressive text generation with KV cache

    Args:
        language: If non-empty, force language by prepending
            ``language <NAME>`` as the assistant prefix.
        stream: If True, print tokens as they are generated.
        model_dtype: Numpy dtype matching the model precision.
    """
    batch_size = 1

    # Step 1: Compute LFR fbank features
    input_features = preprocess_audio(audio).astype(model_dtype)  # (1, T, 560)

    # Step 2: Run audio encoder (includes adaptor → LLM-dim output)
    audio_out = sessions["audio_encoder"].run({"input_features": input_features})
    audio_features = audio_out["audio_features"]  # (1, audio_seq, llm_hidden)
    num_audio_tokens = audio_features.shape[1]

    # Flatten to (num_audio_tokens, llm_hidden) for the embedding model
    audio_features_2d = audio_features.reshape(-1, audio_features.shape[-1])

    # Step 3: Build prompt with Fun-ASR format
    # Fun-ASR uses: system="You are a helpful assistant."
    #   user="语音转写成{language}：" + fake_tokens for audio positions  # noqa: RUF003
    if language:
        user_text = f"语音转写成{language}："  # noqa: RUF001
    else:
        user_text = "语音转写："  # noqa: RUF001

    system_prompt = "You are a helpful assistant."
    chat_prefix = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_text}"
    )
    prefix_ids = tokenizer.encode(chat_prefix, add_special_tokens=False)

    # Audio placeholder tokens — will be overwritten with audio embeddings
    audio_placeholder_ids = [0] * num_audio_tokens

    chat_suffix = "<|im_end|>\n<|im_start|>assistant\n"
    suffix_ids = tokenizer.encode(chat_suffix, add_special_tokens=False)

    prompt_ids = prefix_ids + audio_placeholder_ids + suffix_ids
    input_ids = np.array([prompt_ids], dtype=np.int64)

    # Step 4: Run embedding model to fuse text + audio
    # The ONNX embedding model:
    #   1. Embeds input_ids via embed_tokens
    #   2. Identifies token_id=0 positions (audio placeholders)
    #   3. Replaces those positions with audio_features
    #   4. Outputs fused inputs_embeds
    embed_out = sessions["embedding"].run(
        {"input_ids": input_ids, "audio_features": audio_features_2d}
    )
    inputs_embeds = embed_out["inputs_embeds"]  # (1, seq_len, hidden)

    # Step 5: Autoregressive decoding with the decoder model
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    # Initialize empty KV cache
    past_kv = {}
    for i in range(num_layers):
        past_kv[f"past_key_values.{i}.key"] = np.zeros(
            (batch_size, num_kv_heads, 0, head_dim), dtype=model_dtype
        )
        past_kv[f"past_key_values.{i}.value"] = np.zeros(
            (batch_size, num_kv_heads, 0, head_dim), dtype=model_dtype
        )

    # Prefill pass with fused embeddings
    # Fun-ASR uses standard RoPE (not MRoPE), so position_ids is (1, seq_len)
    prefill_len = inputs_embeds.shape[1]
    position_ids = np.arange(prefill_len, dtype=np.int64)[np.newaxis, :]  # (1, seq_len)

    decoder_feeds = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": np.ones((batch_size, prefill_len), dtype=np.int64),
        "position_ids": position_ids,
        **past_kv,
    }
    out = sessions["decoder"].run(decoder_feeds)

    # Get first generated token
    logits = out["logits"]
    next_token = int(np.argmax(logits[:, -1, :]))
    generated_ids = [next_token]

    # Update KV cache
    for i in range(num_layers):
        past_kv[f"past_key_values.{i}.key"] = out[f"present.{i}.key"]
        past_kv[f"past_key_values.{i}.value"] = out[f"present.{i}.value"]

    past_seq_len = prefill_len

    # Decode loop
    eos_ids = {151643, 151645}  # <|endoftext|>, <|im_end|>
    stream_ids: list[int] = []
    printed_len = 0
    for _ in range(max_new_tokens - 1):
        if next_token in eos_ids:
            break

        # Decode step: embed single token (no audio features)
        cur_ids = np.array([[next_token]], dtype=np.int64)
        dummy_audio = np.zeros((0, audio_features_2d.shape[-1]), dtype=model_dtype)
        embed_out = sessions["embedding"].run(
            {"input_ids": cur_ids, "audio_features": dummy_audio}
        )
        cur_embeds = embed_out["inputs_embeds"]

        total_seq_len = past_seq_len + 1
        position_ids = np.array([[past_seq_len]], dtype=np.int64)  # (1, 1)

        decoder_feeds = {
            "inputs_embeds": cur_embeds,
            "attention_mask": np.ones((batch_size, total_seq_len), dtype=np.int64),
            "position_ids": position_ids,
            **past_kv,
        }
        out = sessions["decoder"].run(decoder_feeds)

        logits = out["logits"]
        next_token = int(np.argmax(logits[:, -1, :]))
        generated_ids.append(next_token)

        # Stream output
        stream_ids.append(next_token)
        if stream:
            full_text = tokenizer.decode(stream_ids, skip_special_tokens=True)
            new_text = full_text[printed_len:]
            safe_end = new_text.find("\ufffd")
            if safe_end == -1:
                if new_text:
                    print(new_text, end="", flush=True)
                    printed_len = len(full_text)
            elif safe_end > 0:
                print(new_text[:safe_end], end="", flush=True)
                printed_len += safe_end

        for i in range(num_layers):
            past_kv[f"past_key_values.{i}.key"] = out[f"present.{i}.key"]
            past_kv[f"past_key_values.{i}.value"] = out[f"present.{i}.value"]
        past_seq_len = total_seq_len

    # Flush remaining buffered characters
    if stream and stream_ids:
        final_text = tokenizer.decode(stream_ids, skip_special_tokens=True)
        remaining = final_text[printed_len:]
        if remaining:
            remaining = remaining.replace("\ufffd", "")
            if remaining:
                print(remaining, end="", flush=True)

    print()
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def transcribe_long(
    sessions,
    tokenizer,
    audio: np.ndarray,
    config,
    *,
    chunk_length: float = 30.0,
    **kwargs,
) -> str:
    """Transcribe audio of any length by chunking.

    Splits audio into segments of ``chunk_length`` seconds and
    transcribes each independently, concatenating the results.
    """
    samples_per_chunk = int(chunk_length * SAMPLE_RATE)
    total_samples = len(audio)

    if total_samples <= samples_per_chunk:
        return transcribe(sessions, tokenizer, audio, config, **kwargs)

    num_chunks = (total_samples + samples_per_chunk - 1) // samples_per_chunk
    results = []
    for i in range(num_chunks):
        start = i * samples_per_chunk
        end = min(start + samples_per_chunk, total_samples)
        chunk = audio[start:end]
        if len(chunk) < SAMPLE_RATE * 0.3:
            continue  # Skip very short trailing chunks
        print(f"\n[Chunk {i + 1}/{num_chunks}] ", end="", flush=True)
        text = transcribe(sessions, tokenizer, chunk, config, **kwargs)
        results.append(text.strip())

    return " ".join(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fun-ASR-Nano speech recognition with ONNX models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help="HuggingFace model ID (default: %(default)s).",
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="Path to an audio file. If omitted, records from mic.",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help=(
            "Force language. Languages: auto, zh, en, ja. Default: auto (model auto-detects)."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "webgpu"],
        help="Execution provider (default: cpu).",
    )
    parser.add_argument(
        "--dtype",
        default="f32",
        choices=["f32", "f16", "bf16"],
        help="Model precision (default: f32).",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output (print all at once).",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Continuously record and transcribe (loop until Ctrl+C).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Maximum tokens to generate per chunk (default: %(default)s).",
    )
    parser.add_argument(
        "--chunk-length",
        type=float,
        default=600.0,
        help="Audio chunk length in seconds for long files (default: 600).",
    )
    parser.add_argument(
        "--save-to",
        metavar="DIR",
        default=None,
        help="Save ONNX models to DIR and exit (no inference).",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Exit with non-zero code on failure (for CI pipelines).",
    )
    args = parser.parse_args()

    # Resolve language
    lang_key = args.language.lower().strip()
    if lang_key not in LANGUAGE_MAP:
        supported = ", ".join(sorted(set(LANGUAGE_MAP.keys()) - {"auto"}))
        parser.error(f"Unknown language {args.language!r}. Supported: auto, {supported}")
    forced_language = LANGUAGE_MAP[lang_key]

    # Build config from model YAML + LLM config
    print(f"Building ONNX models from {args.model!r} (dtype={args.dtype}) ...")
    config = build_fun_asr_config(args.model, dtype=args.dtype)

    # Build the 3 ONNX models using build_from_module
    from mobius.models.fun_asr import FunASRForConditionalGeneration
    from mobius.tasks import FunASRSpeechLanguageTask

    module = FunASRForConditionalGeneration(config)
    pkg = build_from_module(module, config, task=FunASRSpeechLanguageTask())

    if args.save_to:
        pkg.save(args.save_to, check_weights=False)
        print(f"Saved to {args.save_to}")
        return

    # Apply weights from HF checkpoint.
    # Fun-ASR stores weights in model.pt (a single PyTorch checkpoint),
    # not the standard HuggingFace safetensors layout.
    import torch
    from huggingface_hub import hf_hub_download

    weights_path = hf_hub_download(args.model, "model.pt")
    # weights_only=False required: model.pt uses pickle format with nested
    # state_dict structure, not safetensors
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    # model.pt wraps the actual weights in a 'state_dict' key
    state_dict = checkpoint.get("state_dict", checkpoint)
    if hasattr(module, "preprocess_weights"):
        state_dict = module.preprocess_weights(state_dict)
    prefix_map = getattr(module, "weight_prefix_map", None)
    pkg.apply_weights(state_dict, prefix_map=prefix_map)

    # Create ORT sessions for each model
    device = args.device
    print(f"Creating inference sessions (device={device}) ...")
    sessions = {name: OnnxModelSession(model, device=device) for name, model in pkg.items()}

    # Load tokenizer from the Qwen3-0.6B subfolder
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model, subfolder="Qwen3-0.6B", trust_remote_code=True
    )

    print("Ready.\n")
    if forced_language:
        print(f"Language: {forced_language} (forced)")
    else:
        print("Language: auto-detect")

    do_stream = not args.no_stream

    np_dtype_map = {
        "f32": np.float32,
        "f16": np.float16,
        "bf16": ml_dtypes.bfloat16,
    }
    np_dtype = np_dtype_map[args.dtype]

    transcribe_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        language=forced_language,
        stream=do_stream,
        model_dtype=np_dtype,
    )

    def do_transcribe(audio_data):
        return transcribe_long(
            sessions,
            tokenizer,
            audio_data,
            config,
            chunk_length=args.chunk_length,
            **transcribe_kwargs,
        )

    if args.audio:
        print(f"Loading audio: {args.audio}")
        audio = load_audio_file(args.audio)
        print(f"Audio: {len(audio) / SAMPLE_RATE:.1f}s\n")
        text = do_transcribe(audio)
        print(f"\n📝 Result: {text}")
    elif args.continuous:
        print("=== Continuous ASR Mode (Ctrl+C to exit) ===\n")
        try:
            while True:
                audio = record_until_enter()
                if len(audio) < SAMPLE_RATE * 0.5:
                    print("  (too short, skipping)\n")
                    continue
                text = do_transcribe(audio)
                print(f"📝 {text}\n")
        except KeyboardInterrupt:
            print("\nDone.")
    else:
        audio = record_until_enter()
        if len(audio) < SAMPLE_RATE * 0.3:
            print("No audio recorded.")
            return
        text = do_transcribe(audio)
        print(f"\n📝 Result: {text}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if "--ci" in sys.argv:
            print(f"FAILED: {e}", file=sys.stderr)
            sys.exit(1)
        raise
