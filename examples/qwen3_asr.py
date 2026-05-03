#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qwen3-ASR speech recognition with ONNX models.

Builds three ONNX models from ``mobius`` (audio encoder,
embedding, decoder) and runs the full ASR pipeline:

    audio → mel spectrogram → audio encoder → embedding fusion → decoder → text

Supports real-time microphone input and audio file input.

Prerequisites::

    pip install mobius-ai[transformers] sounddevice

Usage::

    # Record from microphone (press Enter to stop)
    python examples/qwen3_asr.py

    # Transcribe an audio file
    python examples/qwen3_asr.py --audio speech.wav

    # Continuous mic mode (Ctrl+C to exit)
    python examples/qwen3_asr.py --continuous

    # Force language detection (useful for dialects)
    # 指定语言（支持方言）
    python examples/qwen3_asr.py --language zh          # Mandarin / 普通话
    python examples/qwen3_asr.py --language yue          # Cantonese / 粤语
    python examples/qwen3_asr.py --language en           # English
    python examples/qwen3_asr.py --language ja           # Japanese
    python examples/qwen3_asr.py --language dongbei      # Dongbei dialect / 东北话
    python examples/qwen3_asr.py --language 四川话        # Sichuan dialect

    # Use a different model size
    python examples/qwen3_asr.py --model Qwen/Qwen3-ASR-1.7B

    # GPU inference with half precision
    python examples/qwen3_asr.py --device cuda --dtype fp16

    # Disable streaming output
    python examples/qwen3_asr.py --no-stream

    # Save ONNX models without running inference
    python examples/qwen3_asr.py --save-to output/qwen3-asr/
"""

from __future__ import annotations

import argparse
import sys
import threading

import numpy as np
import transformers

from mobius import build
from mobius._testing.ort_inference import OnnxModelSession

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
SAMPLE_RATE = 16000
MAX_RECORD_SECONDS = 60
MAX_NEW_TOKENS = 256

# Qwen3-ASR special tokens
AUDIO_START_TOKEN_ID = 151669
AUDIO_TOKEN_ID = 151676  # <|audio_pad|>
AUDIO_END_TOKEN_ID = 151670
ASR_TEXT_TOKEN = 151704  # <asr_text>

# Chat template token IDs (Qwen-style)
IM_START = 151644  # <|im_start|>
IM_END = 151645  # <|im_end|>
SYSTEM_ID = 8948  # "system"
USER_ID = 872  # "user"
ASSISTANT_ID = 77091  # "assistant"
NEWLINE_ID = 198  # "\n"

# Language / dialect mapping.
# Keys are CLI aliases; values are the language names used in the model's
# "language <NAME><asr_text>" generation prefix.
# 语言/方言映射：键为命令行别名，值为模型使用的语言名称。
#
# Supports 30 languages + 22 Chinese dialects.
LANGUAGE_MAP: dict[str, str] = {
    # Auto-detect (default: model decides the language)
    "auto": "",
    # --- Languages (30) ---
    # Chinese / 中文 / 普通话
    "zh": "Chinese",
    "chinese": "Chinese",
    "mandarin": "Chinese",
    "普通话": "Chinese",
    "中文": "Chinese",
    # English / 英语
    "en": "English",
    "english": "English",
    "英语": "English",
    # Japanese / 日语
    "ja": "Japanese",
    "japanese": "Japanese",
    "日语": "Japanese",
    # Korean / 韩语
    "ko": "Korean",
    "korean": "Korean",
    "韩语": "Korean",
    # Arabic / 阿拉伯语
    "ar": "Arabic",
    "arabic": "Arabic",
    "阿拉伯语": "Arabic",
    # German / 德语
    "de": "German",
    "german": "German",
    "德语": "German",
    # French / 法语
    "fr": "French",
    "french": "French",
    "法语": "French",
    # Spanish / 西班牙语
    "es": "Spanish",
    "spanish": "Spanish",
    "西班牙语": "Spanish",
    # Portuguese / 葡萄牙语
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "葡萄牙语": "Portuguese",
    # Indonesian / 印尼语
    "id": "Indonesian",
    "indonesian": "Indonesian",
    "印尼语": "Indonesian",
    # Italian / 意大利语
    "it": "Italian",
    "italian": "Italian",
    "意大利语": "Italian",
    # Russian / 俄语
    "ru": "Russian",
    "russian": "Russian",
    "俄语": "Russian",
    # Thai / 泰语
    "th": "Thai",
    "thai": "Thai",
    "泰语": "Thai",
    # Vietnamese / 越南语
    "vi": "Vietnamese",
    "vietnamese": "Vietnamese",
    "越南语": "Vietnamese",
    # Turkish / 土耳其语
    "tr": "Turkish",
    "turkish": "Turkish",
    "土耳其语": "Turkish",
    # Hindi / 印地语
    "hi": "Hindi",
    "hindi": "Hindi",
    "印地语": "Hindi",
    # Malay / 马来语
    "ms": "Malay",
    "malay": "Malay",
    "马来语": "Malay",
    # Dutch / 荷兰语
    "nl": "Dutch",
    "dutch": "Dutch",
    "荷兰语": "Dutch",
    # Swedish / 瑞典语
    "sv": "Swedish",
    "swedish": "Swedish",
    "瑞典语": "Swedish",
    # Danish / 丹麦语
    "da": "Danish",
    "danish": "Danish",
    "丹麦语": "Danish",
    # Finnish / 芬兰语
    "fi": "Finnish",
    "finnish": "Finnish",
    "芬兰语": "Finnish",
    # Polish / 波兰语
    "pl": "Polish",
    "polish": "Polish",
    "波兰语": "Polish",
    # Czech / 捷克语
    "cs": "Czech",
    "czech": "Czech",
    "捷克语": "Czech",
    # Filipino / 菲律宾语
    "fil": "Filipino",
    "filipino": "Filipino",
    "菲律宾语": "Filipino",
    # Persian / 波斯语
    "fa": "Persian",
    "persian": "Persian",
    "波斯语": "Persian",
    # Greek / 希腊语
    "el": "Greek",
    "greek": "Greek",
    "希腊语": "Greek",
    # Hungarian / 匈牙利语
    "hu": "Hungarian",
    "hungarian": "Hungarian",
    "匈牙利语": "Hungarian",
    # Macedonian / 马其顿语
    "mk": "Macedonian",
    "macedonian": "Macedonian",
    "马其顿语": "Macedonian",
    # Romanian / 罗马尼亚语
    "ro": "Romanian",
    "romanian": "Romanian",
    "罗马尼亚语": "Romanian",
    # --- Chinese Dialects (22) ---
    # Cantonese / 粤语
    "yue": "Cantonese",
    "cantonese": "Cantonese",
    "粤语": "Cantonese",
    # Cantonese-HK / 香港粤语
    "cantonese-hk": "Cantonese-HK",
    "香港粤语": "Cantonese-HK",
    # Cantonese-GD / 广东粤语
    "cantonese-gd": "Cantonese-GD",
    "广东粤语": "Cantonese-GD",
    # Wu / Shanghainese / 吴语 / 上海话
    "wuu": "Shanghainese",
    "wu": "Shanghainese",
    "shanghainese": "Shanghainese",
    "吴语": "Shanghainese",
    "上海话": "Shanghainese",
    # Min Nan / Hokkien / 闽南语
    "minnan": "Minnan",
    "hokkien": "Minnan",
    "闽南语": "Minnan",
    # Hakka / 客家话
    "hakka": "Hakka",
    "客家话": "Hakka",
    # Sichuan / 四川话
    "sichuan": "Sichuan",
    "四川话": "Sichuan",
    # Anhui / 安徽话
    "anhui": "Anhui",
    "安徽话": "Anhui",
    # Dongbei / 东北话
    "dongbei": "Dongbei",
    "东北话": "Dongbei",
    # Fujian / 福建话
    "fujian": "Fujian",
    "福建话": "Fujian",
    # Gansu / 甘肃话
    "gansu": "Gansu",
    "甘肃话": "Gansu",
    # Guizhou / 贵州话
    "guizhou": "Guizhou",
    "贵州话": "Guizhou",
    # Hebei / 河北话
    "hebei": "Hebei",
    "河北话": "Hebei",
    # Henan / 河南话
    "henan": "Henan",
    "河南话": "Henan",
    # Hubei / 湖北话
    "hubei": "Hubei",
    "湖北话": "Hubei",
    # Hunan / 湖南话
    "hunan": "Hunan",
    "湖南话": "Hunan",
    # Jiangxi / 江西话
    "jiangxi": "Jiangxi",
    "江西话": "Jiangxi",
    # Ningxia / 宁夏话
    "ningxia": "Ningxia",
    "宁夏话": "Ningxia",
    # Shandong / 山东话
    "shandong": "Shandong",
    "山东话": "Shandong",
    # Shaanxi / 陕西话
    "shaanxi": "Shaanxi",
    "陕西话": "Shaanxi",
    # Shanxi / 山西话
    "shanxi": "Shanxi",
    "山西话": "Shanxi",
    # Tianjin / 天津话
    "tianjin": "Tianjin",
    "天津话": "Tianjin",
    # Yunnan / 云南话
    "yunnan": "Yunnan",
    "云南话": "Yunnan",
    # Zhejiang / 浙江话
    "zhejiang": "Zhejiang",
    "浙江话": "Zhejiang",
}


# ---------------------------------------------------------------------------
# Mel spectrogram
# ---------------------------------------------------------------------------


def compute_mel_spectrogram(
    audio: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = 128,
    n_fft: int = 400,
    hop_length: int = 160,
) -> np.ndarray:
    """Compute log-mel spectrogram using WhisperFeatureExtractor.

    Returns array of shape ``(1, n_mels, time_frames)``.
    """
    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor(
        feature_size=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        sampling_rate=sample_rate,
    )
    out = fe(
        audio,
        sampling_rate=sample_rate,
        return_tensors="np",
        padding=False,
    )
    return out["input_features"].astype(np.float32)


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
# ASR inference pipeline
# ---------------------------------------------------------------------------


def transcribe(
    sessions: dict[str, OnnxModelSession],
    tokenizer,
    audio: np.ndarray,
    config,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
    language: str = "",
    stream: bool = True,
) -> str:
    """Full ASR pipeline: audio → text.

    Runs three ONNX models in sequence:
    1. Audio encoder: mel spectrogram → audio features
    2. Embedding: fuse text tokens with audio features
    3. Decoder: autoregressive text generation with KV cache

    Args:
        language: If non-empty, force language by prepending
            ``language <NAME><asr_text>`` as the assistant prefix.
        stream: If True, print tokens as they are generated.
    """
    batch_size = 1

    # Step 1: Compute mel spectrogram
    mel = compute_mel_spectrogram(audio)  # (1, n_mels, time)

    # Step 2: Run audio encoder
    audio_out = sessions["audio_encoder"].run({"input_features": mel})
    audio_features = audio_out["audio_features"]  # (1, audio_seq, dim)
    num_audio_tokens = audio_features.shape[1]

    # Flatten to (num_audio_tokens, output_dim) for the embedding model
    audio_features_2d = audio_features.reshape(-1, audio_features.shape[-1])

    # Step 3: Build chat-template prompt with audio token placeholders
    # Format: <|im_start|>system\n<|im_end|>\n
    #         <|im_start|>user\n<|audio_start|><|audio_pad|>*N<|audio_end|>
    #         <|im_end|>\n<|im_start|>assistant\n
    prompt_ids = (
        [
            IM_START,
            SYSTEM_ID,
            NEWLINE_ID,
            IM_END,
            NEWLINE_ID,
            IM_START,
            USER_ID,
            NEWLINE_ID,
            AUDIO_START_TOKEN_ID,
        ]
        + [AUDIO_TOKEN_ID] * num_audio_tokens
        + [AUDIO_END_TOKEN_ID, IM_END, NEWLINE_ID, IM_START, ASSISTANT_ID, NEWLINE_ID]
    )

    # When language is forced, append "language <NAME><asr_text>" tokens
    # as the assistant's initial response to skip language detection.
    if language:
        prefix_text = f"language {language}<asr_text>"
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        prompt_ids.extend(prefix_ids)

    input_ids = np.array([prompt_ids], dtype=np.int64)

    # Step 4: Run embedding model (fuse text + audio)
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
            (batch_size, num_kv_heads, 0, head_dim), dtype=np.float32
        )
        past_kv[f"past_key_values.{i}.value"] = np.zeros(
            (batch_size, num_kv_heads, 0, head_dim), dtype=np.float32
        )

    # Prefill pass with fused embeddings
    prefill_len = inputs_embeds.shape[1]
    pos = np.arange(prefill_len, dtype=np.int64)[np.newaxis, :]
    # MRoPE: all 3 dims get same positions for text-only generation
    position_ids = np.stack([pos, pos, pos])  # (3, 1, seq_len)

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

    # Decode loop: feed each new token back through embedding + decoder
    eos_ids = {151643, 151645}  # <|endoftext|>, <|im_end|>
    streaming = False  # Start streaming after <asr_text>
    for _ in range(max_new_tokens - 1):
        if next_token in eos_ids:
            break

        # For decode steps, use embedding with single token
        # (no audio features — zeros since there are no audio tokens)
        cur_ids = np.array([[next_token]], dtype=np.int64)
        dummy_audio = np.zeros((0, audio_features_2d.shape[-1]), dtype=np.float32)
        embed_out = sessions["embedding"].run(
            {"input_ids": cur_ids, "audio_features": dummy_audio}
        )
        cur_embeds = embed_out["inputs_embeds"]

        total_seq_len = past_seq_len + 1
        pos = np.array([[past_seq_len]], dtype=np.int64)
        position_ids = np.stack([pos, pos, pos])  # (3, 1, 1)

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

        # Stream output (skip prefix tokens before <asr_text>)
        if next_token == ASR_TEXT_TOKEN:
            streaming = True
        elif streaming and stream:
            text = tokenizer.decode([next_token], skip_special_tokens=True)
            print(text, end="", flush=True)

        for i in range(num_layers):
            past_kv[f"past_key_values.{i}.key"] = out[f"present.{i}.key"]
            past_kv[f"past_key_values.{i}.value"] = out[f"present.{i}.value"]
        past_seq_len = total_seq_len

    print()
    raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return parse_asr_output(raw)


def parse_asr_output(raw: str) -> str:
    """Strip ``language X<asr_text>`` prefix from raw ASR output."""
    import re

    m = re.match(r"language\s+\w+<asr_text>", raw)
    if m:
        return raw[m.end() :]
    return raw


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Qwen3-ASR speech recognition with ONNX models.\nQwen3-ASR 语音识别（ONNX 模型）。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help="HuggingFace model ID / 模型 ID (default: %(default)s).",
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="Path to an audio file / 音频文件路径. If omitted, records from mic.",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help=(
            "Force language/dialect / 强制指定语言或方言. "
            "Languages: auto, zh, en, ja, ko, ar, de, fr, es, pt, id, it, "
            "ru, th, vi, tr, hi, ms, nl, sv, da, fi, pl, cs, fil, fa, el, "
            "hu, mk, ro. "
            "Dialects: yue, wuu, sichuan, minnan, hakka, anhui, dongbei, "
            "fujian, gansu, guizhou, hebei, henan, hubei, hunan, jiangxi, "
            "ningxia, shandong, shaanxi, shanxi, tianjin, yunnan, zhejiang. "
            "Chinese aliases also accepted (普通话, 粤语, 四川话, etc). "
            "Default: auto (model auto-detects)."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "webgpu"],
        help="Execution provider / 推理设备 (default: cpu).",
    )
    parser.add_argument(
        "--dtype",
        default="fp32",
        choices=["fp32", "fp16"],
        help="Model precision / 模型精度 (default: fp32).",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output / 禁用流式输出 (print all at once).",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Continuously record and transcribe / 连续识别模式 (loop until Ctrl+C).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Maximum tokens to generate / 最大生成令牌数 (default: %(default)s).",
    )
    parser.add_argument(
        "--save-to",
        metavar="DIR",
        default=None,
        help="Save ONNX models to DIR and exit / 保存模型到目录 (no inference).",
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
    forced_language = LANGUAGE_MAP[lang_key]  # Empty string for "auto"

    # Map dtype flag to mobius dtype string
    dtype_map = {"fp32": "f32", "fp16": "f16"}
    dtype = dtype_map[args.dtype]

    # Build the 3 ONNX models (auto-detected from model_type)
    print(f"Building ONNX models from {args.model!r} (dtype={args.dtype}) ...")
    pkg = build(args.model, dtype=dtype, load_weights=not args.save_to)
    config = pkg.config

    if args.save_to:
        pkg.save(args.save_to, check_weights=False)
        print(f"Saved to {args.save_to}")
        return

    # Create ORT sessions for each model.
    device = args.device
    print(f"Creating inference sessions (device={device}) ...")
    sessions = {name: OnnxModelSession(model, device=device) for name, model in pkg.items()}

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Ready.\n")
    if forced_language:
        print(f"Language: {forced_language} (forced)")
    else:
        print("Language: auto-detect")

    stream = not args.no_stream

    def do_transcribe(audio_data):
        return transcribe(
            sessions,
            tokenizer,
            audio_data,
            config,
            max_new_tokens=args.max_new_tokens,
            language=forced_language,
            stream=stream,
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
