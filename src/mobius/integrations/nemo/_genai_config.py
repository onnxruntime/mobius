# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""ONNX Runtime GenAI bundle generation for NeMo FastConformer-RNNT models.

The :func:`write_genai_bundle` helper writes a directory that the ONNX Runtime
GenAI ``nemotron_speech`` C++ pipeline can load directly:

* ``encoder.onnx`` / ``decoder.onnx`` / ``joint.onnx`` — the streaming encoder,
  RNN-T prediction network and joiner, saved flat with GenAI tensor layouts
  (see :mod:`mobius.tasks._rnnt`).
* ``genai_config.json`` — model type, architecture/mel parameters and the
  logical→ONNX tensor-name mappings the runtime reads.
* ``audio_processor_config.json`` — log-mel front-end parameters.
* ``tokenizer.json`` / ``tokenizer_config.json`` — the SentencePiece vocabulary
  converted to the HuggingFace Unigram form ORT-Extensions loads via the
  ``T5Tokenizer`` path.
* ``silero_vad.onnx`` (optional) — Silero voice-activity-detection model used by
  the streaming processor to drop prolonged-silence chunks.

The GenAI runtime hardcodes float32 encoder I/O, so the package must be built
with ``dtype="f32"`` (the default).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import onnx_ir as ir

from mobius._model_package import ModelPackage
from mobius.integrations.nemo._reader import NeMoArchive

# NeMo's AudioToMelSpectrogramPreprocessor log zero-guard (2**-24).
_LOG_EPS = 5.96046448e-08
_SILERO_REPO = "onnx-community/silero-vad"
_SILERO_FILE = "onnx/model.onnx"
# Pin a specific revision so include_vad bundles are reproducible rather than
# shipping whatever mutable HEAD the repo currently points at.
_SILERO_REVISION = "e71cae966052b992a7eca6b17738916ce0eca4ec"

# Logical (GenAI) tensor name -> mobius ONNX tensor name, per sub-model. The
# GenAI ``nemotron_speech`` pipeline reads these mappings from genai_config.json.
_ENCODER_INPUTS = {
    "audio_features": "audio_signal",
    "input_lengths": "length",
    "cache_last_channel": "cache_last_channel",
    "cache_last_time": "cache_last_time",
    "cache_last_channel_len": "cache_last_channel_len",
}
_ENCODER_OUTPUTS = {
    "encoder_outputs": "encoder_output",
    "output_lengths": "encoder_length",
    "cache_last_channel_next": "cache_last_channel_next",
    "cache_last_time_next": "cache_last_time_next",
    "cache_last_channel_len_next": "cache_last_channel_len_next",
}
_DECODER_INPUTS = {
    "targets": "targets",
    "lstm_hidden_state": "state_h",
    "lstm_cell_state": "state_c",
}
_DECODER_OUTPUTS = {
    "outputs": "decoder_output",
    "lstm_hidden_state": "state_h_out",
    "lstm_cell_state": "state_c_out",
}
_JOINER_INPUTS = {
    "encoder_outputs": "encoder_outputs",
    "decoder_outputs": "decoder_outputs",
}
_JOINER_OUTPUTS = {"logits": "logits"}


def _preproc_window_samples(value: Any, sample_rate: int, default: int) -> int:
    """Resolve a NeMo window size/stride (seconds or samples) to sample count."""
    if isinstance(value, float) and value < 1.0:
        return round(value * sample_rate)
    if isinstance(value, int):
        return value
    return default


def _derive_params(config: dict[str, Any], chunk_seconds: float) -> dict[str, Any]:
    """Derive GenAI config scalars from a NeMo ``model_config.yaml`` dict."""
    preproc = config.get("preprocessor", {}) or {}
    encoder = config.get("encoder", {}) or {}
    joint = config.get("joint", {}) or {}
    decoder = config.get("decoder", {}) or {}
    prednet = decoder.get("prednet", {}) or {}

    sample_rate = int(preproc.get("sample_rate", 16000))
    num_mels = int(preproc.get("features", preproc.get("nfilt", 128)))
    n_fft = int(preproc.get("n_fft", 512))
    preemph = preproc.get("preemph", 0.97)
    preemph = 0.97 if preemph is None else float(preemph)

    # win/hop are stored in seconds for this preprocessor; convert to samples.
    win_length = _preproc_window_samples(
        preproc.get("window_size", preproc.get("n_window_size")), sample_rate, 400
    )
    hop_length = _preproc_window_samples(
        preproc.get("window_stride", preproc.get("n_window_stride")), sample_rate, 160
    )

    # vocab (with blank) = num_classes + 1; blank is the final index.
    num_classes = int(joint.get("num_classes", decoder.get("vocab_size", 1024)))
    vocab_size = num_classes + 1
    blank_id = num_classes

    subsampling_factor = int(encoder.get("subsampling_factor", 8))
    # att_context_size rows are [left, right]; NeMo may store a single row or a
    # list of rows (cache-aware multi-context training). The largest (primary)
    # left context drives the running attention cache size.
    att_context = encoder.get("att_context_size", [70, 13])
    if att_context and isinstance(att_context[0], (list, tuple)):
        att_context = att_context[0]
    left_context = int(att_context[0])
    conv_context = int(encoder.get("conv_kernel_size", 9)) - 1
    pre_encode_cache_size = encoder.get("pre_encode_cache_size")
    if isinstance(pre_encode_cache_size, (list, tuple)):
        pre_encode_cache_size = pre_encode_cache_size[-1]
    if not pre_encode_cache_size:
        # NeMo default for an 8x dw_striding subsampling stem.
        pre_encode_cache_size = 9
    pre_encode_cache_size = int(pre_encode_cache_size)

    max_symbols = int(
        ((config.get("decoding", {}) or {}).get("greedy", {}) or {}).get("max_symbols", 10)
    )

    return {
        "vocab_size": vocab_size,
        "blank_id": blank_id,
        "num_mels": num_mels,
        "fft_size": n_fft,
        "hop_length": hop_length,
        "win_length": win_length,
        "preemph": preemph,
        "sample_rate": sample_rate,
        "subsampling_factor": subsampling_factor,
        "left_context": left_context,
        "conv_context": conv_context,
        "pre_encode_cache_size": pre_encode_cache_size,
        "chunk_samples": round(chunk_seconds * sample_rate),
        "max_symbols_per_step": max_symbols,
        "hidden_size": int(encoder.get("d_model", 1024)),
        "num_hidden_layers": int(encoder.get("n_layers", 24)),
        "decoder_hidden": int(prednet.get("pred_hidden", 640)),
        "decoder_layers": int(prednet.get("pred_rnn_layers", 2)),
        "dither": float(preproc.get("dither", 0.0) or 0.0),
        "normalize": preproc.get("normalize", "none"),
    }


def _save_flat(model: ir.Model, path: Path) -> None:
    """Serialize a single model as ``<name>.onnx`` + ``<name>.onnx.data``."""
    ir.save(model, str(path), external_data=f"{path.name}.data")


def _build_tokenizer_files(archive: NeMoArchive, dest: Path) -> int:
    """Write HuggingFace Unigram ``tokenizer.json`` from the SentencePiece model.

    Returns the vocabulary size including the appended ``<blank>`` symbol.
    """
    import sentencepiece as spm

    tmp = dest / "_spm"
    written = archive.extract_tokenizer(tmp)
    model_path = written.get("model_path")
    if model_path is None:
        raise ValueError("NeMo archive has no SentencePiece tokenizer model_path")
    sp = spm.SentencePieceProcessor(model_file=model_path)

    # Token list ordered by id; the RNN-T blank occupies the final vocab slot.
    tokens = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    tokens.append("<blank>")

    # Unigram vocab as [token, score]; rank-based scores (lower id = higher).
    unigram_vocab = [
        [tok, 0.0 if tok in ("<unk>", "<blank>") else -float(i)]
        for i, tok in enumerate(tokens)
    ]
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": 0,
                "content": "<unk>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        ],
        "normalizer": {
            "type": "Replace",
            "pattern": {"String": " "},
            "content": "\u2581",
        },
        "pre_tokenizer": {
            "type": "Metaspace",
            "replacement": "\u2581",
            "add_prefix_space": True,
        },
        "post_processor": None,
        # Metaspace decoder converts the ▁ word-boundary marks back to spaces.
        "decoder": {
            "type": "Metaspace",
            "replacement": "\u2581",
            "add_prefix_space": True,
        },
        "model": {"type": "Unigram", "unk_id": 0, "vocab": unigram_vocab},
    }
    (dest / "tokenizer.json").write_text(
        json.dumps(tokenizer_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tokenizer_config = {
        "tokenizer_class": "T5Tokenizer",
        "unk_token": "<unk>",
        "model_max_length": 1024,
        "add_bos_token": False,
        "add_eos_token": False,
        "clean_up_tokenization_spaces": False,
    }
    (dest / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Clean up the extracted SentencePiece scratch files.
    for f in tmp.glob("*"):
        f.unlink()
    tmp.rmdir()
    return len(tokens)


def _download_silero_vad(dest: Path) -> bool:
    """Best-effort download of the Silero VAD ONNX model. Returns success."""
    try:
        import shutil

        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(
            repo_id=_SILERO_REPO, filename=_SILERO_FILE, revision=_SILERO_REVISION
        )
        shutil.copy2(cached, str(dest / "silero_vad.onnx"))
    except Exception:  # pragma: no cover - network/availability dependent
        return False
    else:
        return True


def write_genai_bundle(
    pkg: ModelPackage,
    archive: NeMoArchive,
    dest_dir: str | Path,
    *,
    chunk_seconds: float = 1.12,
    include_vad: bool = True,
) -> Path:
    """Write an ONNX Runtime GenAI ``nemotron_speech`` bundle.

    Args:
        pkg: The :class:`~mobius._model_package.ModelPackage` produced by
            :func:`~mobius.integrations.nemo.build_from_nemo`. Must contain the
            ``encoder_streaming``, ``decoder`` and ``joint`` graphs.
        archive: The source :class:`NeMoArchive` (for preprocessor parameters
            and the SentencePiece tokenizer).
        dest_dir: Output directory (created if needed).
        chunk_seconds: Streaming chunk length in seconds. The default ``1.12``
            matches the model's native ``att_context_size`` ``[70, 13]`` chunk.
        include_vad: When ``True`` (default), best-effort download of the Silero
            VAD model and add its config block. If the download fails the bundle
            is still written without VAD.

    Returns:
        The resolved output directory path.
    """
    for key in ("encoder_streaming", "decoder", "joint"):
        if key not in pkg:
            raise KeyError(f"ModelPackage is missing required model {key!r}")

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    # Flat ONNX files named per the GenAI pipeline's defaults.
    _save_flat(pkg["encoder_streaming"], dest / "encoder.onnx")
    _save_flat(pkg["decoder"], dest / "decoder.onnx")
    _save_flat(pkg["joint"], dest / "joint.onnx")

    params = _derive_params(archive.config, chunk_seconds)

    encoder_block = {
        "filename": "encoder.onnx",
        "hidden_size": params["hidden_size"],
        "num_hidden_layers": params["num_hidden_layers"],
        "inputs": dict(_ENCODER_INPUTS),
        "outputs": dict(_ENCODER_OUTPUTS),
    }
    decoder_block = {
        "filename": "decoder.onnx",
        "hidden_size": params["decoder_hidden"],
        "num_hidden_layers": params["decoder_layers"],
        "inputs": dict(_DECODER_INPUTS),
        "outputs": dict(_DECODER_OUTPUTS),
    }
    joiner_block = {
        "filename": "joint.onnx",
        "inputs": dict(_JOINER_INPUTS),
        "outputs": dict(_JOINER_OUTPUTS),
    }

    model_block: dict[str, Any] = {
        "type": "nemotron_speech",
        "vocab_size": params["vocab_size"],
        "num_mels": params["num_mels"],
        "fft_size": params["fft_size"],
        "hop_length": params["hop_length"],
        "win_length": params["win_length"],
        "preemph": params["preemph"],
        "log_eps": _LOG_EPS,
        "subsampling_factor": params["subsampling_factor"],
        "left_context": params["left_context"],
        "conv_context": params["conv_context"],
        "pre_encode_cache_size": params["pre_encode_cache_size"],
        "sample_rate": params["sample_rate"],
        "chunk_samples": params["chunk_samples"],
        "blank_id": params["blank_id"],
        "max_symbols_per_step": params["max_symbols_per_step"],
        "encoder": encoder_block,
        "decoder": decoder_block,
        "joiner": joiner_block,
    }

    if include_vad and _download_silero_vad(dest):
        model_block["vad"] = {
            "filename": "silero_vad.onnx",
            "threshold": 0.3,
            "silence_duration_ms": 3360,
            "prefix_padding_ms": 560,
        }

    genai_config = {"model": model_block}
    (dest / "genai_config.json").write_text(
        json.dumps(genai_config, indent=2), encoding="utf-8"
    )

    audio_config = {
        "model_type": "speech_features",
        "audio_params": {
            "sample_rate": params["sample_rate"],
            "n_fft": params["fft_size"],
            "hop_length": params["hop_length"],
            "n_mels": params["num_mels"],
            "window_length": params["win_length"],
            "window_type": "hann",
            "fmin": 0,
            "fmax": params["sample_rate"] // 2,
            "dither": params["dither"],
            "preemphasis": params["preemph"],
            "log_zero_guard_type": "add",
            "log_zero_guard_value": 1e-10,
            "normalize": params["normalize"],
            "center": True,
            "mag_power": 2.0,
        },
    }
    (dest / "audio_processor_config.json").write_text(
        json.dumps(audio_config, indent=2), encoding="utf-8"
    )

    _build_tokenizer_files(archive, dest)
    return dest
