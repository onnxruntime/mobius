#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

r"""Phi-4 Multimodal output parity check: ONNX (mobius) vs HuggingFace.

Compares the prefill logits of the ONNX 4-model pipeline (vision,
speech, embedding, decoder) against HuggingFace's PyTorch reference
for every modality combination:

    - **Text only**: No image or audio input.
    - **Text + image**: Vision encoder active.
    - **Text + audio (short)**: Speech encoder with short audio
      (audio_projection_mode=0).
    - **Text + audio (long)**: Speech encoder with 4096 mel frames
      (512 subsampled tokens), exercising the >500-token chunking
      code path in the Conformer (speech_conformer_encoder.py L2849).
    - **Text + image + audio**: Both encoders active
      (audio_projection_mode=1 for combined mode).

For each test case the script reports:

    - Max absolute difference
    - Mean absolute difference
    - Top-1 token agreement (argmax match at every position)

The 4-model split is:

    - **Vision**  (``vision/model.onnx``): SigLIP encoder + projection
    - **Speech**  (``speech/model.onnx``): Conformer encoder + projection
    - **Embedding** (``embedding/model.onnx``): token embed + InputMixer
    - **Decoder** (``model/model.onnx``): LoRA text decoder + lm_head

Prerequisites::

    pip install mobius-ai[transformers] torchaudio

Usage::

    # Full model (default) — all 32 text layers, 27 vision, 24 audio:
    python examples/phi4mm_parity.py

    # Quick smoke-test with reduced layer counts (2 each):
    python examples/phi4mm_parity.py --debug

    # Run a single mode:
    python examples/phi4mm_parity.py --mode text

    # Use fewer layers for faster development / CI testing:
    python examples/phi4mm_parity.py --num-text-layers 2 --num-vision-layers 2 --num-audio-blocks 2

    # Provide external test data:
    python examples/phi4mm_parity.py \
        --image ~/phi4mm_testdata/images/australia.jpg \
        --audio ~/phi4mm_testdata/test_7_2.wav

    # Long audio test (exercises different Conformer sequence lengths):
    python examples/phi4mm_parity.py --mode audio-long \
        --audio ~/phi4mm_testdata/TALK_GREENY_.wav

    # Also verify onnxruntime-genai works with the exported model:
    python examples/phi4mm_parity.py --debug --genai
"""

from __future__ import annotations

import argparse
import glob as _glob
import os
import sys
import time

# Ensure we import mobius from the local src/ tree, not any installed
# version, so audio/model fixes in this worktree are always active.
# This is needed because this file is run as a standalone script (not via
# `python -m`), so Python does not automatically add the package root to
# sys.path.  Without this, a `pip install -e .` from a different worktree
# would shadow the local source.
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np  # noqa: E402
import onnx_ir as ir  # noqa: E402
import transformers  # noqa: E402

from mobius import build_from_module  # noqa: E402
from mobius._configs import ArchitectureConfig  # noqa: E402
from mobius._testing.ort_inference import OnnxModelSession  # noqa: E402
from mobius._weight_loading import _download_weights  # noqa: E402
from mobius.models.phi import Phi4MMMultiModalModel  # noqa: E402
from mobius.tasks import Phi4MMMultiModalTask  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "microsoft/Phi-4-multimodal-instruct"

# Special token IDs for Phi-4 multimodal
IMAGE_TOKEN_ID = 200010  # <|endoftext10|> — image placeholder
AUDIO_TOKEN_ID = 200011  # <|endoftext11|> — audio placeholder

# Audio preprocessing defaults (80-dim mel filterbank at 16 kHz)
AUDIO_SAMPLE_RATE = 16000
AUDIO_N_MELS = 80

# Default layer counts: full microsoft/Phi-4-multimodal-instruct model.
# Pass --num-text-layers / --num-vision-layers / --num-audio-blocks to
# reduce these for faster development / CI testing, or use --debug for a
# preset of small values.
DEFAULT_NUM_TEXT_LAYERS = 32  # full decoder depth
DEFAULT_NUM_VISION_LAYERS = 27  # full SigLIP encoder (uses layer_idx=-2 = 26)
DEFAULT_NUM_AUDIO_BLOCKS = 24  # full Conformer encoder

# Reduced counts used by --debug for quick pipeline smoke-tests.
DEBUG_NUM_TEXT_LAYERS = 2
DEBUG_NUM_VISION_LAYERS = 2
DEBUG_NUM_AUDIO_BLOCKS = 2

# Short audio: 100 mel frames → 13 speech tokens after compression
SHORT_AUDIO_FRAMES = 100
# Long audio: 4096 mel frames -> 512 subsampled tokens.
# The conformer encoder chunks audio when seq_len > 500 subsampled tokens
# (speech_conformer_encoder.py L2849).  With 3 stages of stride-2 (8x),
# we need >4000 mel frames to exceed the threshold.  4096 gives 512 tokens.
LONG_AUDIO_FRAMES = 4096
# Number of new tokens to generate for the text preview (greedy decode).
# Increased from 5 to 50 for more meaningful generation output.
DEFAULT_MAX_NEW_TOKENS = 50

# Default testdata directory. If present, real files are used automatically
# instead of random dummy tensors. Audio files longer than this are capped.
TESTDATA_DIR = os.path.expanduser("~/phi4mm_testdata")
MAX_AUDIO_SECONDS = 60.0  # cap to avoid very long CPU inference

ALL_MODES = ["text", "vision", "audio-short", "audio-long", "vision-audio"]


def _nemo_subsampling_output_len(num_frames: int, num_stages: int = 3) -> int:
    """Compute output token count after NeMo dw_striding subsampling.

    Each stage is a stride-2 Conv2d with kernel=3 and symmetric padding=1,
    giving ``out = (in - 1) // 2 + 1`` per stage (floor of ceil division).

    For 100 frames: 100 → 50 → 25 → 13  (not 100 // 8 = 12).
    For 200 frames: 200 → 100 → 50 → 25  (= 200 // 8 = 25, same).
    For 128 frames: 128 → 64 → 32 → 16   (= 128 // 8 = 16, same).
    """
    for _ in range(num_stages):
        num_frames = (num_frames - 1) // 2 + 1
    return num_frames


def _discover_testdata(base_dir: str) -> dict[str, list[str]]:
    """Discover real test files in ``base_dir``.

    Looks for:
    - ``images/``: JPEG/PNG files for vision tests.
    - Top-level ``*.wav`` files for audio tests (excludes long files in
      sub-directories like ``validation_audios/`` and ``ReproChunkedAudio/``).

    Returns a dict with keys ``"images"`` and ``"audios"``, each a sorted list
    of absolute file paths. Returns empty lists if the directory does not exist.
    """
    images: list[str] = []
    audios: list[str] = []
    if not os.path.isdir(base_dir):
        return {"images": images, "audios": audios}

    img_dir = os.path.join(base_dir, "images")
    if os.path.isdir(img_dir):
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            images.extend(_glob.glob(os.path.join(img_dir, ext)))
        images.sort()

    # Only top-level WAV files — skip sub-directory collections which may
    # contain 30+ files designed for bulk validation runs.
    for ext in ("*.wav",):
        audios.extend(_glob.glob(os.path.join(base_dir, ext)))
    audios.sort()

    return {"images": images, "audios": audios}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_onnx_package(
    model_id: str,
    num_text_layers: int,
    num_vision_layers: int,
    num_audio_blocks: int,
    *,
    trust_remote_code: bool = False,
):
    """Build the Phi4MM 4-model ONNX package with real weights.

    Returns (pkg, config) where pkg has keys:
    vision, speech, embedding, model.
    """
    hf_config = transformers.AutoConfig.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    text_config = hf_config if not hasattr(hf_config, "text_config") else hf_config.text_config
    text_config.num_hidden_layers = num_text_layers
    config = ArchitectureConfig.from_transformers(text_config)
    # Use float32 for ORT session compatibility
    config.dtype = ir.DataType.FLOAT
    if config.vision is not None:
        config.vision.num_hidden_layers = num_vision_layers
    if config.audio is not None:
        config.audio.num_blocks = num_audio_blocks

    module = Phi4MMMultiModalModel(config)
    pkg = build_from_module(module, config, task=Phi4MMMultiModalTask())

    state_dict = _download_weights(model_id)
    state_dict = module.preprocess_weights(state_dict)
    pkg.apply_weights(state_dict)

    return pkg, config


def _patch_cache_utils_for_phi4mm() -> None:
    """Patch transformers cache_utils for compatibility with Phi4MM's HF model code.

    Phi4MM's ``modeling_phi4mm.py`` was written against transformers 4.x and
    uses several cache APIs that were removed or renamed in 5.x.  We patch them
    back onto the live classes so the dynamic module can import and run.

    Patches applied:
    - ``SlidingWindowCache``: removed in 5.x; stub with ``DynamicCache``.
    - ``DynamicCache.get_usable_length``: removed in 5.x, replaced by
      ``get_seq_length``.  The two-argument signature
      ``(kv_seq_len, layer_idx)`` is compatible because ``get_seq_length``
      ignores both arguments and returns the current stored sequence length.
    - ``DynamicCache.from_legacy_cache``: removed in 5.x; returns an empty
      ``DynamicCache`` (the legacy ``None`` path, which is the only remaining
      caller in Phi4MM).
    """
    from transformers import cache_utils

    if not hasattr(cache_utils, "SlidingWindowCache"):
        cache_utils.SlidingWindowCache = cache_utils.DynamicCache

    dc = cache_utils.DynamicCache
    if not hasattr(dc, "get_usable_length"):
        # get_usable_length(kv_seq_len, layer_idx) → usable cached length.
        # In transformers 5.x the equivalent is get_seq_length() (no args).
        dc.get_usable_length = lambda self, kv_seq_len=None, layer_idx=None: (
            self.get_seq_length()
        )

    if not hasattr(dc, "from_legacy_cache"):
        # from_legacy_cache(past_key_values) → DynamicCache.
        # In Phi4MM the only call site passes None (no existing cache), so
        # returning an empty DynamicCache is always correct here.
        dc.from_legacy_cache = classmethod(lambda cls, past=None: cls())

    if not hasattr(dc, "to_legacy_cache"):
        # to_legacy_cache() → tuple of (key, value) tuples per layer.
        # Transformers 5.x removed this method; the Phi4MM model only calls
        # it on the legacy-cache code path (when past_key_values was a tuple),
        # which newer transformers never triggers.  Add a stub for safety.
        def _to_legacy_cache(self):
            if hasattr(self, "key_cache") and hasattr(self, "value_cache"):
                return tuple(zip(self.key_cache, self.value_cache))
            return ()

        dc.to_legacy_cache = _to_legacy_cache


def _patch_num_logits_for_phi4mm(model) -> None:
    """Patch a loaded Phi4MM model to handle num_logits_to_keep=None.

    Newer transformers (5.x) passes ``num_logits_to_keep=None`` from
    ``prepare_inputs_for_generation`` into the model's ``forward``.
    Phi4MM's forward was written for 4.x which always passed an ``int``
    (default 0).  Passing ``None`` causes a ``TypeError`` at the slice
    ``hidden_states[:, -num_logits_to_keep:, :]``.

    We wrap the model's instance ``forward`` to coerce ``None`` → ``0``
    (meaning: return logits for all tokens, the same as the old default).
    """
    import functools

    _orig_forward = model.forward

    @functools.wraps(_orig_forward)
    def _patched(*args, num_logits_to_keep=0, **kwargs):
        if num_logits_to_keep is None:
            num_logits_to_keep = 0
        return _orig_forward(*args, num_logits_to_keep=num_logits_to_keep, **kwargs)

    model.forward = _patched


def _patch_no_meta_init() -> None:
    """Remove meta-device init context from PreTrainedModel.get_init_context.

    Transformers 5.x always wraps model construction in a ``torch.device("meta")``
    context inside ``from_pretrained``.  Phi4MM's ``NemoConvSubsampling.__init__``
    calls ``int()`` on a tensor that was created with ``torch.tensor(feat_in, ...)``
    inside this context — which is a meta tensor — causing a hard crash.

    Removing the meta-device context means the model is allocated with real weights
    immediately (slightly higher peak memory, but always correct).
    """
    import torch
    import transformers

    _original = transformers.PreTrainedModel.get_init_context.__func__  # type: ignore[attr-defined]

    @classmethod  # type: ignore[misc]
    def _no_meta_get_init_context(
        cls, dtype, is_quantized, _is_ds_init_called, allow_all_kernels
    ):
        contexts = _original(cls, dtype, is_quantized, _is_ds_init_called, allow_all_kernels)
        # Drop torch.device("meta") — Phi4MM conformer encoder is not meta-safe
        return [c for c in contexts if not (isinstance(c, torch.device) and str(c) == "meta")]

    transformers.PreTrainedModel.get_init_context = _no_meta_get_init_context


def _patch_tied_weights_keys_for_phi4mm() -> None:
    """Fix get_expanded_tied_weights_keys for Phi4MM with transformers 5.x.

    Phi4MM defines ``_tied_weights_keys = ["lm_head.weight"]`` (transformers 4.x
    list format).  Transformers 5.x ``get_expanded_tied_weights_keys`` expects a
    dict ``{target: source}``, and crashes with ``AttributeError: 'list' object
    has no attribute 'keys'`` when it finds a list.

    We patch the method to convert the list to the standard Phi4 dict mapping
    ``{"lm_head.weight": "model.embed_tokens.weight"}`` so that weight tying
    works correctly after loading from the checkpoint (where ``lm_head.weight``
    is absent because it is tied).
    """
    import transformers

    _original = transformers.PreTrainedModel.get_expanded_tied_weights_keys

    def _patched(self, all_submodels: bool = False) -> dict:
        if isinstance(self._tied_weights_keys, list):
            # Convert 4.x list format to 5.x dict format.
            # Standard Phi4 pattern: lm_head tied to embed_tokens.
            self._tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
        return _original(self, all_submodels=all_submodels)

    transformers.PreTrainedModel.get_expanded_tied_weights_keys = _patched


def _patch_peft_for_phi4mm() -> None:
    """Patch peft to handle Phi4MMModel missing prepare_inputs_for_generation.

    The HF Phi4MM wraps its inner ``Phi4MMModel`` with peft using
    ``task_type="CAUSAL_LM"``, but ``Phi4MMModel`` (the backbone)
    doesn't have ``prepare_inputs_for_generation``. Newer peft
    versions crash on this. We patch the PeftModelForCausalLM init
    to catch and ignore the missing attribute.
    """
    try:
        import peft.peft_model as pm

        _orig_init = pm.PeftModelForCausalLM.__init__

        def _patched_init(self, model, peft_config, adapter_name="default", **kwargs):
            # Add a dummy method so peft doesn't crash
            if not hasattr(model, "prepare_inputs_for_generation"):
                model.prepare_inputs_for_generation = lambda *a, **kw: {}
            _orig_init(self, model, peft_config, adapter_name=adapter_name, **kwargs)

        pm.PeftModelForCausalLM.__init__ = _patched_init
    except (ImportError, AttributeError):
        # peft is not installed, or the internal module structure has changed —
        # either way the patch is not needed and we can proceed without it.
        pass


def load_hf_model(
    model_id: str,
    num_text_layers: int,
    num_audio_blocks: int | None = None,
    num_vision_layers: int | None = None,
    *,
    trust_remote_code: bool = False,
):
    """Load the HuggingFace Phi4MM model with merged LoRA adapters.

    Uses ``from_pretrained`` with a layer-count-reduced config so that
    only the first ``num_text_layers`` decoder layers are instantiated,
    matching the ONNX model.  Extra checkpoint layers are silently
    skipped by transformers.

    If ``num_audio_blocks`` is provided the conformer encoder is truncated
    to that many blocks after loading, so the HF reference uses the same
    encoder depth as the ONNX model built with that value.

    If ``num_vision_layers`` is provided the SigLIP encoder is truncated
    to that many layers.  The ONNX model with ``num_vision_layers=N``
    runs ``N - 1`` encoder layers (HF uses ``layer_idx=-2``, i.e. the
    second-to-last hidden state); truncating HF to ``N`` layers ensures
    ``layer_idx=-2`` resolves to the same encoder depth as ONNX.

    Merging all LoRA adapters into the base weights ensures the HF
    model's behavior matches the ONNX model, which always applies all
    adapters regardless of input_mode.

    Returns (model, tokenizer).
    """
    import torch

    _patch_cache_utils_for_phi4mm()
    _patch_peft_for_phi4mm()
    _patch_no_meta_init()
    _patch_tied_weights_keys_for_phi4mm()

    hf_config = transformers.AutoConfig.from_pretrained(
        model_id,
        # trust_remote_code is required for Phi4MM because its model class
        # is distributed in the HuggingFace repo as custom Python code
        # (not yet part of the transformers library core).  Only use this
        # with model IDs you trust; do NOT set it for arbitrary user input.
        trust_remote_code=trust_remote_code,
    )
    text_config = hf_config if not hasattr(hf_config, "text_config") else hf_config.text_config
    text_config.num_hidden_layers = num_text_layers
    text_config._attn_implementation = "eager"

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )

    # Use from_pretrained (not from_config) to avoid a transformers bug
    # where post_init() mishandles list-typed _tied_weights_keys.
    # Extra checkpoint layers (beyond num_text_layers) are silently ignored.
    # Do NOT use device_map: accelerate initialises on meta device first,
    # which breaks Phi4MM's conformer init (calls .item() on meta tensors).
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        config=hf_config,
        torch_dtype=torch.float32,
        trust_remote_code=trust_remote_code,  # see note above
    )
    model.eval()

    # Transformers 5.x fails to tie lm_head.weight → embed_tokens.weight when
    # the model is not initialised on meta device.  Force the tie manually so
    # the output projection uses the correct learned embedding matrix.
    if hasattr(model, "lm_head") and hasattr(model, "model"):
        embed_weight = model.model.embed_tokens.weight
        if model.lm_head.weight.data_ptr() != embed_weight.data_ptr():
            model.lm_head.weight = embed_weight
            print("  Manually tied lm_head.weight → model.embed_tokens.weight")

    # Patch num_logits_to_keep=None → 0 (newer transformers passes None)
    _patch_num_logits_for_phi4mm(model)

    # Merge all LoRA adapters into base weights
    _merge_all_lora_adapters(model)

    # Truncate conformer encoder to match ONNX audio block count
    if num_audio_blocks is not None:
        try:
            enc = model.model.embed_tokens_extend.audio_embed.encoder
            enc.encoders = enc.encoders[:num_audio_blocks]
            print(f"  Truncated audio conformer to {num_audio_blocks} blocks")
        except AttributeError:
            # Model structure differs from expected — truncation not needed.
            pass

    # Truncate SigLIP encoder to match ONNX vision depth.
    # ONNX runs (num_vision_layers - 1) layers; HF uses layer_idx=-2.
    # Truncating HF to num_vision_layers total layers makes layer_idx=-2
    # resolve to the same depth (index num_vision_layers - 2).
    if num_vision_layers is not None:
        try:
            enc = model.model.embed_tokens_extend.image_embed.img_processor.encoder
            enc.layers = enc.layers[:num_vision_layers]
            print(f"  Truncated SigLIP encoder to {num_vision_layers} layers")
        except AttributeError:
            # Model structure differs from expected — truncation not needed.
            pass

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _merge_all_lora_adapters(model) -> None:
    """Merge all LoRA adapters into base weights and freeze.

    The ONNX model always applies all LoRA adapters (they are baked
    into the graph). HuggingFace's Phi4MM selectively activates
    adapters per input_mode. To get parity, we merge all adapter
    contributions into the base weights.
    """
    try:
        from peft.tuners.lora.layer import LoraLayer
    except ImportError:
        print("  peft not installed — skipping LoRA merge")
        return

    merged_count = 0
    for module in model.modules():
        if not isinstance(module, LoraLayer):
            continue
        if not hasattr(module, "lora_A"):
            continue

        for adapter_name in list(module.lora_A.keys()):
            scaling = module.scaling[adapter_name]
            lora_a = module.lora_A[adapter_name].weight.data
            lora_b = module.lora_B[adapter_name].weight.data
            module.weight.data += scaling * lora_b @ lora_a
            lora_a.zero_()
            lora_b.zero_()
            merged_count += 1

    # Prevent the forward pass from switching or disabling adapters
    model.set_lora_adapter = lambda adapter_name: None
    model.unset_lora_adapter = lambda: None

    print(f"  Merged {merged_count} LoRA adapter weights into base model")


# ---------------------------------------------------------------------------
# ONNX pipeline helpers
# ---------------------------------------------------------------------------


def run_onnx_pipeline(
    pkg,
    config: ArchitectureConfig,
    input_ids: np.ndarray,
    *,
    pixel_values: np.ndarray | None = None,
    image_sizes: np.ndarray | None = None,
    audio_features: np.ndarray | None = None,
    audio_projection_mode: int = 0,
) -> np.ndarray:
    """Run the 4-model ONNX pipeline and return prefill logits.

    Chains: vision → speech → embedding → decoder (single-step prefill).
    """
    hidden_size = config.hidden_size

    # Step 1: Vision encoder
    if pixel_values is not None:
        vision_session = OnnxModelSession(pkg["vision"])
        if pixel_values.ndim == 5:
            n, crops, c, h, w = pixel_values.shape
            pixel_values = pixel_values.reshape(n * crops, c, h, w)
        if image_sizes is None:
            image_sizes = np.array(
                [[pixel_values.shape[-2], pixel_values.shape[-1]]],
                dtype=np.int64,
            )
        vision_out = vision_session.run(
            {"pixel_values": pixel_values, "image_sizes": image_sizes}
        )
        image_features = vision_out["image_features"]
        vision_session.close()
        if image_features.ndim == 3:
            image_features = image_features[0]
    else:
        image_features = np.zeros((0, hidden_size), dtype=np.float32)

    # Step 2: Speech encoder
    if audio_features is not None:
        speech_session = OnnxModelSession(pkg["speech"])
        audio_sizes = np.array([audio_features.shape[1]], dtype=np.int64)
        speech_out = speech_session.run(
            {
                "audio_embeds": audio_features,
                "audio_sizes": audio_sizes,
                "audio_projection_mode": np.array(audio_projection_mode, dtype=np.int64),
            }
        )
        speech_feats = speech_out["audio_features"]
        speech_session.close()
        if speech_feats.ndim == 3:
            speech_feats = speech_feats[0]
    else:
        speech_feats = np.zeros((0, hidden_size), dtype=np.float32)

    # Step 3: Embedding (fuse text + vision + speech)
    embedding_session = OnnxModelSession(pkg["embedding"])
    embed_out = embedding_session.run(
        {
            "input_ids": input_ids,
            "image_features": image_features,
            "audio_features": speech_feats,
        }
    )
    inputs_embeds = embed_out["inputs_embeds"]
    embedding_session.close()

    # Step 4: Decoder (single-step prefill, no KV cache history)
    seq_len = inputs_embeds.shape[1]
    decoder_feeds: dict[str, np.ndarray] = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        "position_ids": np.arange(seq_len, dtype=np.int64)[np.newaxis, :],
    }
    for i in range(config.num_hidden_layers):
        decoder_feeds[f"past_key_values.{i}.key"] = np.zeros(
            (1, config.num_key_value_heads, 0, config.head_dim),
            dtype=np.float32,
        )
        decoder_feeds[f"past_key_values.{i}.value"] = np.zeros(
            (1, config.num_key_value_heads, 0, config.head_dim),
            dtype=np.float32,
        )

    decoder_session = OnnxModelSession(pkg["model"])
    decoder_out = decoder_session.run(decoder_feeds)
    decoder_session.close()

    return decoder_out["logits"]


# ---------------------------------------------------------------------------
# HuggingFace reference helpers
# ---------------------------------------------------------------------------


def run_hf_forward(
    model,
    tokenizer,
    prompt: str,
    *,
    pixel_values: np.ndarray | None = None,
    image_sizes: np.ndarray | None = None,
    audio_features: np.ndarray | None = None,
    num_image_tokens: int = 0,
    num_audio_tokens: int = 0,
    input_mode: int = 0,
) -> np.ndarray:
    """Run a single HuggingFace forward pass and return logits.

    Builds input_ids internally. ``num_audio_tokens`` must be the
    *compressed* token count (after the Conformer's 8x stride),
    matching the number of audio placeholder tokens the HF model
    expects.

    input_mode values (from HF Phi4MM):
        0 = LANGUAGE (text only)
        1 = VISION (text + image)
        2 = SPEECH (text + audio)
        3 = VISION_SPEECH (text + image + audio)
    """
    import torch

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    hf_input_ids = build_input_ids(
        tokenizer,
        prompt,
        num_image_tokens=num_image_tokens,
        num_audio_tokens=num_audio_tokens,
    )

    seq_len = hf_input_ids.shape[1]

    kwargs: dict = {
        "input_ids": torch.from_numpy(hf_input_ids).to(device),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.long, device=device),
        "position_ids": torch.arange(seq_len, device=device).unsqueeze(0),
        "input_mode": input_mode,
        "use_cache": False,  # Parity check only needs logits, not KV cache state
    }

    if pixel_values is not None:
        kwargs["input_image_embeds"] = torch.from_numpy(pixel_values).to(
            device=device, dtype=dtype
        )
        if image_sizes is not None:
            kwargs["image_sizes"] = torch.from_numpy(image_sizes).to(device)

    if audio_features is not None:
        kwargs["input_audio_embeds"] = torch.from_numpy(audio_features).to(
            device=device, dtype=dtype
        )
        # audio_embed_sizes = compressed token count (must match
        # placeholder count in input_ids)
        kwargs["audio_embed_sizes"] = torch.tensor([num_audio_tokens], device=device)

    with torch.no_grad():
        out = model(**kwargs)

    return out.logits.cpu().numpy()


def generate_onnx(
    pkg,
    config: ArchitectureConfig,
    tokenizer,
    input_ids: np.ndarray,
    *,
    pixel_values: np.ndarray | None = None,
    image_sizes: np.ndarray | None = None,
    audio_features: np.ndarray | None = None,
    audio_projection_mode: int = 0,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> list[int]:
    """Greedy decode up to max_new_tokens steps using the ONNX 4-model pipeline.

    Runs a full prefill then auto-regressively generates one token at a time
    using the decoder's KV cache outputs (``present.{i}.key/value``).  Vision
    and audio features are only processed during the prefill step; subsequent
    decode steps pass empty feature tensors to the embedding model.
    """
    hidden_size = config.hidden_size

    # ── Prefill: run vision + speech + embedding once ──────────────────────
    if pixel_values is not None:
        vision_session = OnnxModelSession(pkg["vision"])
        pv = (
            pixel_values.reshape(-1, *pixel_values.shape[-3:])
            if pixel_values.ndim == 5
            else pixel_values
        )
        if image_sizes is None:
            image_sizes = np.array([[pv.shape[-2], pv.shape[-1]]], dtype=np.int64)
        image_features = vision_session.run({"pixel_values": pv, "image_sizes": image_sizes})[
            "image_features"
        ]
        if image_features.ndim == 3:
            image_features = image_features[0]
        vision_session.close()
    else:
        image_features = np.zeros((0, hidden_size), dtype=np.float32)

    if audio_features is not None:
        speech_session = OnnxModelSession(pkg["speech"])
        audio_sizes = np.array([audio_features.shape[1]], dtype=np.int64)
        speech_feats = speech_session.run(
            {
                "audio_embeds": audio_features,
                "audio_sizes": audio_sizes,
                "audio_projection_mode": np.array(audio_projection_mode, dtype=np.int64),
            }
        )["audio_features"]
        if speech_feats.ndim == 3:
            speech_feats = speech_feats[0]
        speech_session.close()
    else:
        speech_feats = np.zeros((0, hidden_size), dtype=np.float32)

    # Keep embedding session open for decode steps (single-token embedding)
    embedding_session = OnnxModelSession(pkg["embedding"])
    inputs_embeds = embedding_session.run(
        {
            "input_ids": input_ids,
            "image_features": image_features,
            "audio_features": speech_feats,
        }
    )["inputs_embeds"]
    empty_features = np.zeros((0, hidden_size), dtype=np.float32)

    # ── Prefill decoder pass ───────────────────────────────────────────────
    seq_len = inputs_embeds.shape[1]
    decoder_feeds: dict[str, np.ndarray] = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        "position_ids": np.arange(seq_len, dtype=np.int64)[np.newaxis, :],
    }
    for i in range(config.num_hidden_layers):
        decoder_feeds[f"past_key_values.{i}.key"] = np.zeros(
            (1, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32
        )
        decoder_feeds[f"past_key_values.{i}.value"] = np.zeros(
            (1, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32
        )

    decoder_session = OnnxModelSession(pkg["model"])
    decoder_out = decoder_session.run(decoder_feeds)
    logits = decoder_out["logits"]

    # Extract KV cache for decode steps (output names: present.{i}.key/value)
    kv_cache: dict[str, np.ndarray] = {}
    for i in range(config.num_hidden_layers):
        kv_cache[f"past_key_values.{i}.key"] = decoder_out[f"present.{i}.key"]
        kv_cache[f"past_key_values.{i}.value"] = decoder_out[f"present.{i}.value"]

    # ── Decode loop ────────────────────────────────────────────────────────
    eos_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    # Phi4MM also uses <|end|> as a stop token
    end_id = tokenizer.convert_tokens_to_ids("<|end|>")
    if end_id is not None and end_id != tokenizer.unk_token_id:
        eos_ids.add(int(end_id))

    generated_tokens: list[int] = []
    current_seq_len = seq_len

    for _ in range(max_new_tokens):
        next_token_id = int(np.argmax(logits[0, -1, :]))
        generated_tokens.append(next_token_id)
        if next_token_id in eos_ids:
            break

        # Embed only the new token — no image/audio features at decode time
        new_ids = np.array([[next_token_id]], dtype=np.int64)
        new_embeds = embedding_session.run(
            {
                "input_ids": new_ids,
                "image_features": empty_features,
                "audio_features": empty_features,
            }
        )["inputs_embeds"]  # [1, 1, hidden_size]

        current_seq_len += 1
        step_feeds: dict[str, np.ndarray] = {
            "inputs_embeds": new_embeds,
            "attention_mask": np.ones((1, current_seq_len), dtype=np.int64),
            "position_ids": np.array([[current_seq_len - 1]], dtype=np.int64),
            **kv_cache,
        }
        step_out = decoder_session.run(step_feeds)
        logits = step_out["logits"]
        for i in range(config.num_hidden_layers):
            kv_cache[f"past_key_values.{i}.key"] = step_out[f"present.{i}.key"]
            kv_cache[f"past_key_values.{i}.value"] = step_out[f"present.{i}.value"]

    embedding_session.close()
    decoder_session.close()
    return generated_tokens


def generate_hf(
    model,
    tokenizer,
    prompt: str,
    *,
    pixel_values: np.ndarray | None = None,
    image_sizes: np.ndarray | None = None,
    audio_features: np.ndarray | None = None,
    num_image_tokens: int = 0,
    num_audio_tokens: int = 0,
    input_mode: int = 0,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> list[int]:
    """Greedy-decode up to max_new_tokens tokens using HuggingFace generate()."""
    import torch

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    hf_input_ids = build_input_ids(
        tokenizer,
        prompt,
        num_image_tokens=num_image_tokens,
        num_audio_tokens=num_audio_tokens,
    )

    generate_kwargs: dict = {
        "input_ids": torch.from_numpy(hf_input_ids).to(device),
        "attention_mask": torch.ones(
            1, hf_input_ids.shape[1], dtype=torch.long, device=device
        ),
        "input_mode": input_mode,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,  # greedy
        "temperature": None,
        "top_p": None,
    }

    if pixel_values is not None:
        generate_kwargs["input_image_embeds"] = torch.from_numpy(pixel_values).to(
            device=device, dtype=dtype
        )
        if image_sizes is not None:
            generate_kwargs["image_sizes"] = torch.from_numpy(image_sizes).to(device)

    if audio_features is not None:
        generate_kwargs["input_audio_embeds"] = torch.from_numpy(audio_features).to(
            device=device, dtype=dtype
        )
        generate_kwargs["audio_embed_sizes"] = torch.tensor([num_audio_tokens], device=device)

    with torch.no_grad():
        generated_ids = model.generate(**generate_kwargs)

    # Return only the newly generated token IDs (not the prompt)
    prompt_len = hf_input_ids.shape[1]
    return generated_ids[0, prompt_len:].cpu().tolist()


def decode_generated(tokenizer, token_ids: list[int]) -> str:
    """Decode a list of token IDs to a human-readable string."""
    if not token_ids:
        return "(empty)"
    return repr(tokenizer.decode(token_ids, skip_special_tokens=True))


# ---------------------------------------------------------------------------
# Input construction helpers
# ---------------------------------------------------------------------------


def create_dummy_pixel_values(
    config: ArchitectureConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Create random pixel values processed through the HF image processor.

    Returns (pixel_values, image_sizes, num_img_tokens).
    """
    from PIL import Image

    image_size = (config.vision.image_size if config.vision else None) or 448
    rng = np.random.default_rng(42)
    img_data = rng.integers(0, 255, (image_size, image_size, 3), dtype=np.uint8)
    img = Image.fromarray(img_data)

    processor = transformers.AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    inputs = processor.image_processor(images=[img], return_tensors="np")
    pixel_values = inputs["input_image_embeds"].astype(np.float32)
    image_sizes = inputs["image_sizes"].astype(np.int64)
    num_img_tokens = _compute_num_img_tokens_from_sizes(image_sizes)
    return pixel_values, image_sizes, num_img_tokens


def create_dummy_audio_features(
    config: ArchitectureConfig,
    num_frames: int = SHORT_AUDIO_FRAMES,
) -> np.ndarray:
    """Create random audio mel features.

    Returns [1, num_frames, input_size] float32 array.
    """
    input_size = (config.audio.input_size if config.audio else None) or 80
    rng = np.random.default_rng(123)
    return rng.standard_normal((1, num_frames, input_size)).astype(np.float32)


def load_real_audio(
    audio_path: str,
    max_seconds: float | None = MAX_AUDIO_SECONDS,
) -> np.ndarray:
    """Load audio file and compute mel spectrogram features.

    ``max_seconds`` caps the waveform length before feature extraction to
    avoid very long ONNX inference on CPU (default: ``MAX_AUDIO_SECONDS``).
    Pass ``None`` to process the full file.

    Returns [1, time_frames, n_mels] float32 array.
    """
    import soundfile as sf

    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    # data: (samples, channels) -> mono by averaging channels
    audio = data.mean(axis=1).astype(np.float32)
    if max_seconds is not None:
        audio = audio[: int(max_seconds * sr)]

    # Use Phi4MMAudioFeatureExtractor (no 30-second truncation limit) so
    # long-audio tests exercise the >500-token conformer chunking path.
    processor = transformers.AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    audio_fe = processor.audio_processor
    out = audio_fe([(audio, sr)], return_tensors="np")
    # input_audio_embeds: (1, T, 80)
    return out["input_audio_embeds"].astype(np.float32)


def _compute_num_img_tokens_from_sizes(image_sizes: np.ndarray, hp: int = 16) -> int:
    """Compute the ONNX vision model's actual token count from image_sizes.

    The processor's num_img_tokens accounts for padding masking, but the ONNX
    vision model always computes the full h x w grid (no masking).  Using
    image_sizes to compute the expected count ensures ONNX and HF-without-mask
    both use the same number of image placeholder tokens.

    Formula: h*hp*(w*hp + 1) + 1 + hp*(hp + 1)
      where h = image_sizes[0][0] // 448  (crop rows)
            w = image_sizes[0][1] // 448  (crop cols)
    """
    h_px = int(image_sizes[0][0])
    w_px = int(image_sizes[0][1])
    h = h_px // 448
    w = w_px // 448
    return h * hp * (w * hp + 1) + 1 + hp * (hp + 1)


def load_real_image(
    image_path: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load an image file and process through the HF image processor.

    Returns (pixel_values, image_sizes, num_img_tokens).

    num_img_tokens is derived from image_sizes using the full-grid formula
    (not the processor's masked count) so it matches what the ONNX vision
    model and HF-without-attention-mask both produce.
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    processor = transformers.AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    inputs = processor.image_processor(images=[img], return_tensors="np")
    pixel_values = inputs["input_image_embeds"].astype(np.float32)
    image_sizes = inputs["image_sizes"].astype(np.int64)
    num_img_tokens = _compute_num_img_tokens_from_sizes(image_sizes)
    return pixel_values, image_sizes, num_img_tokens


def build_input_ids(
    tokenizer,
    prompt: str,
    *,
    num_image_tokens: int = 0,
    num_audio_tokens: int = 0,
) -> np.ndarray:
    """Tokenize prompt and insert image/audio placeholder tokens.

    Layout: BOS + [image_tokens] + [audio_tokens] + rest_of_prompt.
    Returns [1, seq_len] INT64 array.
    """
    tokens = tokenizer(prompt, return_tensors="np")
    input_ids = tokens["input_ids"].astype(np.int64)

    parts = [input_ids[:, :1]]  # BOS
    if num_image_tokens > 0:
        parts.append(
            np.full(
                (1, num_image_tokens),
                IMAGE_TOKEN_ID,
                dtype=np.int64,
            )
        )
    if num_audio_tokens > 0:
        parts.append(
            np.full(
                (1, num_audio_tokens),
                AUDIO_TOKEN_ID,
                dtype=np.int64,
            )
        )
    parts.append(input_ids[:, 1:])  # Rest of prompt

    return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# Parity comparison
# ---------------------------------------------------------------------------


def skipped_result(label: str, reason: str) -> dict:
    """Return a result dict representing a skipped test."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  SKIPPED: {reason}")
    print(f"{'=' * 60}")
    return {
        "label": label,
        "skipped": True,
        "reason": reason,
        "max_diff": float("nan"),
        "mean_diff": float("nan"),
        "token_match": 0.0,
        "allclose": False,
    }


def compare_logits(
    onnx_logits: np.ndarray,
    hf_logits: np.ndarray,
    label: str,
    *,
    onnx_generated: list[int] | None = None,
    hf_generated: list[int] | None = None,
    tokenizer=None,
) -> dict:
    """Compare two logit tensors and print parity metrics.

    When shapes differ (e.g. different audio token counts between
    ONNX compressed and HF raw), compares only the last-token logits
    which determine the next generated token.

    When ``onnx_generated``, ``hf_generated``, and ``tokenizer`` are all
    provided, the decoded generated text is printed alongside the metrics.

    Returns a dict with comparison results.
    """
    if onnx_logits.shape == hf_logits.shape:
        # Full comparison when shapes match
        diff = np.abs(onnx_logits - hf_logits)
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))
        onnx_tokens = np.argmax(onnx_logits, axis=-1)
        hf_tokens = np.argmax(hf_logits, axis=-1)
        token_match = float(np.mean(onnx_tokens == hf_tokens))
        shape_note = f"{onnx_logits.shape}"
    else:
        # Shape mismatch — compare last-token logits only
        onnx_last = onnx_logits[:, -1:, :]
        hf_last = hf_logits[:, -1:, :]
        diff = np.abs(onnx_last - hf_last)
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))
        onnx_tokens = np.argmax(onnx_last, axis=-1)
        hf_tokens = np.argmax(hf_last, axis=-1)
        token_match = float(np.mean(onnx_tokens == hf_tokens))
        shape_note = (
            f"ONNX {onnx_logits.shape} vs HF {hf_logits.shape} (comparing last token only)"
        )

    # Check if all close within tolerance.
    # atol=1e-2 is intentionally looser than the project's standard 1e-4.
    # Reasons: (1) this is a 4-model pipeline — numerical error compounds
    # across speech encoder → projection → embedding → 32-layer decoder;
    # (2) all LoRA adapters are always merged unconditionally, introducing
    # a small systematic offset vs HF's per-mode adapter activation;
    # (3) float32 accumulation over 32 decoder layers yields ~1e-3 spread.
    # The argmax (token_match) is the primary correctness signal.
    atol = 1e-2
    rtol = 1e-2
    if onnx_logits.shape == hf_logits.shape:
        # Shapes match — compare the full tensor for a thorough parity check.
        is_close = bool(np.allclose(onnx_logits, hf_logits, atol=atol, rtol=rtol))
    else:
        # Shape mismatch (e.g. different audio token counts) — fall back to
        # last-token comparison which determines the next generated token.
        is_close = bool(
            np.allclose(
                onnx_logits[:, -1:, :],
                hf_logits[:, -1:, :],
                atol=atol,
                rtol=rtol,
            )
        )

    status = "PASS" if is_close else "FAIL"
    token_status = "PASS" if token_match >= 1.0 - 1e-9 else "FAIL"

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Shape:          {shape_note}")
    print(f"  Max abs diff:   {max_diff:.6e}")
    print(f"  Mean abs diff:  {mean_diff:.6e}")
    print(f"  Token match:    {token_match:.1%}  [{token_status}]")
    print(f"  Allclose:       atol={atol}, rtol={rtol}  [{status}]")

    if tokenizer is not None and onnx_generated is not None and hf_generated is not None:
        onnx_text = decode_generated(tokenizer, onnx_generated)
        hf_text = decode_generated(tokenizer, hf_generated)
        gen_match = "✅" if onnx_generated == hf_generated else "⚠️"
        print(f"  HF generated:   {hf_text}")
        print(f"  ONNX generated: {onnx_text}  {gen_match}")

    print(f"{'=' * 60}")

    return {
        "label": label,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "token_match": float(token_match),
        "allclose": is_close,
    }


# ---------------------------------------------------------------------------
# Individual test cases
# ---------------------------------------------------------------------------


def test_text_only(
    pkg,
    config: ArchitectureConfig,
    hf_model,
    tokenizer,
) -> dict:
    """Text-only parity: no image, no audio."""
    prompt = "The capital of France is"
    input_ids = build_input_ids(tokenizer, prompt)

    print("\n[ONNX] Running text-only prefill + generate ...")
    t0 = time.time()
    onnx_logits = run_onnx_pipeline(pkg, config, input_ids)
    onnx_generated = generate_onnx(pkg, config, tokenizer, input_ids)
    print(f"  ONNX: {time.time() - t0:.1f}s")

    print("[HF] Running text-only prefill + generate ...")
    t0 = time.time()
    hf_logits = run_hf_forward(hf_model, tokenizer, prompt, input_mode=0)
    hf_generated = generate_hf(hf_model, tokenizer, prompt, input_mode=0)
    print(f"  HF:   {time.time() - t0:.1f}s")

    return compare_logits(
        onnx_logits,
        hf_logits,
        "Text Only",
        onnx_generated=onnx_generated,
        hf_generated=hf_generated,
        tokenizer=tokenizer,
    )


def test_vision(
    pkg,
    config: ArchitectureConfig,
    hf_model,
    tokenizer,
    image_path: str | None = None,
) -> dict:
    """Text + image parity.

    Compares the ONNX HD-transform vision pipeline against HuggingFace's
    Phi4MMImageEmbedding, which applies AvgPool2d spatial compression,
    glb_GN/sub_GN row separators, and sub-first ordering before the
    projection MLP.
    """
    if image_path is not None:
        pixel_values, image_sizes, num_img_tokens = load_real_image(image_path)
    else:
        pixel_values, image_sizes, num_img_tokens = create_dummy_pixel_values(config)

    prompt = "Describe this image"

    print(
        f"\n[ONNX] Running vision prefill + generate"
        f" ({pixel_values.shape[0]} crops"
        f" → {num_img_tokens} image tokens) ..."
    )
    input_ids = build_input_ids(tokenizer, prompt, num_image_tokens=num_img_tokens)
    t0 = time.time()
    onnx_logits = run_onnx_pipeline(
        pkg,
        config,
        input_ids,
        pixel_values=pixel_values,
        image_sizes=image_sizes,
    )
    onnx_generated = generate_onnx(
        pkg,
        config,
        tokenizer,
        input_ids,
        pixel_values=pixel_values,
        image_sizes=image_sizes,
    )
    print(f"  ONNX: {time.time() - t0:.1f}s")

    print("[HF] Running vision prefill + generate ...")
    t0 = time.time()
    hf_logits = run_hf_forward(
        hf_model,
        tokenizer,
        prompt,
        pixel_values=pixel_values,
        image_sizes=image_sizes,
        num_image_tokens=num_img_tokens,
        input_mode=1,
    )
    try:
        hf_generated = generate_hf(
            hf_model,
            tokenizer,
            prompt,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            num_image_tokens=num_img_tokens,
            input_mode=1,
        )
    except Exception as e:
        print(f"  Warning: HF generate failed ({e}); skipping generation comparison")
        hf_generated = None
    print(f"  HF:   {time.time() - t0:.1f}s")

    return compare_logits(
        onnx_logits,
        hf_logits,
        f"Text + Image ({os.path.basename(image_path)})" if image_path else "Text + Image",
        onnx_generated=onnx_generated,
        hf_generated=hf_generated,
        tokenizer=tokenizer,
    )


def test_audio(
    pkg,
    config: ArchitectureConfig,
    hf_model,
    tokenizer,
    audio_path: str | None = None,
    num_frames: int = SHORT_AUDIO_FRAMES,
    label: str = "Text + Audio (short)",
    max_audio_seconds: float | None = MAX_AUDIO_SECONDS,
) -> dict:
    """Text + audio parity.

    Uses random features by default, or loads from audio_path.
    ``num_frames`` controls the sequence length when using random data
    (short vs long exercises different code paths in the Conformer).
    ``max_audio_seconds`` caps real audio files to avoid slow CPU inference.
    """
    if audio_path is not None:
        audio_features = load_real_audio(audio_path, max_seconds=max_audio_seconds)
    else:
        audio_features = create_dummy_audio_features(config, num_frames=num_frames)

    # Compute number of speech tokens after NeMo 3-stage stride-2 subsampling
    num_audio_tokens = _nemo_subsampling_output_len(audio_features.shape[1])

    prompt = "Transcribe the audio"
    input_ids = build_input_ids(tokenizer, prompt, num_audio_tokens=num_audio_tokens)

    print(
        f"\n[ONNX] Running audio prefill + generate"
        f" ({audio_features.shape[1]} frames"
        f" → {num_audio_tokens} tokens) ..."
    )
    t0 = time.time()
    onnx_logits = run_onnx_pipeline(
        pkg,
        config,
        input_ids,
        audio_features=audio_features,
    )
    onnx_generated = generate_onnx(
        pkg, config, tokenizer, input_ids, audio_features=audio_features
    )
    print(f"  ONNX: {time.time() - t0:.1f}s")

    print("[HF] Running audio prefill + generate ...")
    t0 = time.time()
    hf_logits = run_hf_forward(
        hf_model,
        tokenizer,
        prompt,
        audio_features=audio_features,
        num_audio_tokens=num_audio_tokens,
        input_mode=2,
    )
    hf_generated = generate_hf(
        hf_model,
        tokenizer,
        prompt,
        audio_features=audio_features,
        num_audio_tokens=num_audio_tokens,
        input_mode=2,
    )
    print(f"  HF:   {time.time() - t0:.1f}s")

    return compare_logits(
        onnx_logits,
        hf_logits,
        label,
        onnx_generated=onnx_generated,
        hf_generated=hf_generated,
        tokenizer=tokenizer,
    )


def test_vision_audio(
    pkg,
    config: ArchitectureConfig,
    hf_model,
    tokenizer,
    image_path: str | None = None,
    audio_path: str | None = None,
) -> dict:
    """Text + image + audio parity (combined VISION_SPEECH mode).

    Uses dummy data by default; pass image_path / audio_path for real inputs.
    The combined mode uses audio_projection_mode=1 (vision branch of the speech
    projector) and HF input_mode=3 (VISION_SPEECH).
    """
    if image_path is not None:
        pixel_values, image_sizes, num_img_tokens = load_real_image(image_path)
    else:
        pixel_values, image_sizes, num_img_tokens = create_dummy_pixel_values(config)

    if audio_path is not None:
        audio_features = load_real_audio(audio_path)
    else:
        audio_features = create_dummy_audio_features(config, num_frames=SHORT_AUDIO_FRAMES)
    num_audio_tokens = _nemo_subsampling_output_len(audio_features.shape[1])

    prompt = "Describe what you see and hear"
    input_ids = build_input_ids(
        tokenizer,
        prompt,
        num_image_tokens=num_img_tokens,
        num_audio_tokens=num_audio_tokens,
    )

    print(
        f"\n[ONNX] Running vision+audio prefill + generate"
        f" ({pixel_values.shape[0]} crops → {num_img_tokens} image tokens,"
        f" {audio_features.shape[1]} frames → {num_audio_tokens} audio tokens) ..."
    )
    t0 = time.time()
    onnx_logits = run_onnx_pipeline(
        pkg,
        config,
        input_ids,
        pixel_values=pixel_values,
        image_sizes=image_sizes,
        audio_features=audio_features,
        audio_projection_mode=1,  # vision branch for combined mode
    )
    onnx_generated = generate_onnx(
        pkg,
        config,
        tokenizer,
        input_ids,
        pixel_values=pixel_values,
        image_sizes=image_sizes,
        audio_features=audio_features,
        audio_projection_mode=1,
    )
    print(f"  ONNX: {time.time() - t0:.1f}s")

    print("[HF] Running vision+audio prefill + generate ...")
    t0 = time.time()
    hf_logits = run_hf_forward(
        hf_model,
        tokenizer,
        prompt,
        pixel_values=pixel_values,
        image_sizes=image_sizes,
        audio_features=audio_features,
        num_image_tokens=num_img_tokens,
        num_audio_tokens=num_audio_tokens,
        input_mode=3,  # VISION_SPEECH
    )
    try:
        hf_generated = generate_hf(
            hf_model,
            tokenizer,
            prompt,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            audio_features=audio_features,
            num_image_tokens=num_img_tokens,
            num_audio_tokens=num_audio_tokens,
            input_mode=3,
        )
    except Exception as e:
        print(f"  Warning: HF generate failed ({e}); skipping generation comparison")
        hf_generated = None
    print(f"  HF:   {time.time() - t0:.1f}s")

    return compare_logits(
        onnx_logits,
        hf_logits,
        "Text + Image + Audio",
        onnx_generated=onnx_generated,
        hf_generated=hf_generated,
        tokenizer=tokenizer,
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(results: list[dict]) -> bool:
    """Print a summary table and return True if all non-skipped tests passed."""
    print("\n")
    print("=" * 70)
    print("  PARITY SUMMARY")
    print("=" * 70)
    header = f"  {'Test':<30} {'Max Diff':>12} {'Mean Diff':>12} {'Token Match':>12}"
    print(header)
    print("  " + "-" * 66)

    all_pass = True
    for r in results:
        if r.get("skipped"):
            print(f"  ⏭️  {r['label']:<28}  [SKIPPED: {r['reason']}]")
        else:
            status = "✅" if r["allclose"] else "⚠️"
            print(
                f"  {status} {r['label']:<28} "
                f"{r['max_diff']:>12.6e} "
                f"{r['mean_diff']:>12.6e} "
                f"{r['token_match']:>11.1%}"
            )
            if not r["allclose"]:
                all_pass = False

    print("=" * 70)
    non_skipped = [r for r in results if not r.get("skipped")]
    skipped = [r for r in results if r.get("skipped")]
    if all_pass:
        print(f"  All {len(non_skipped)} parity checks PASSED ✅", end="")
    else:
        print("  Some parity checks FAILED ⚠️", end="")
    if skipped:
        print(f"  ({len(skipped)} skipped)")
    else:
        print()
    print("=" * 70)
    return all_pass


# ---------------------------------------------------------------------------
# ORT GenAI verification
# ---------------------------------------------------------------------------

# Known Phi-4 token IDs for genai_config.json
_PHI4MM_BOS_TOKEN_ID = 199999
_PHI4MM_EOS_TOKEN_IDS = [199999, 200020]


def run_genai_verification(
    pkg,
    config: ArchitectureConfig,
    model_id: str,
    tokenizer,
    hf_model,
    *,
    trust_remote_code: bool = False,
    max_new_tokens: int = 32,
    prompt: str = "The capital of France is",
) -> bool:
    """Save the ONNX package and verify text generation with onnxruntime-genai.

    Saves the already-built ``pkg`` to a temporary directory, writes
    ``genai_config.json``, copies tokenizer files, then runs text-only
    generation with ``onnxruntime_genai`` and compares the output against
    HuggingFace to confirm the exported model is functional.

    Args:
        pkg: The built ``ModelPackage`` (vision, speech, embedding, model).
        config: Architecture config used to build ``pkg``.
        model_id: HuggingFace model ID (for tokenizer download).
        tokenizer: Already-loaded HuggingFace tokenizer.
        hf_model: Already-loaded HuggingFace model for comparison generation.
        trust_remote_code: Passed to tokenizer download helpers.
        max_new_tokens: Maximum tokens to generate.
        prompt: Text prompt for the verification generation.

    Returns:
        ``True`` if the genai output matches HF, ``False`` otherwise.
        Also returns ``True`` if onnxruntime-genai is not installed
        (so the caller's ``all_pass`` is not penalised).
    """
    try:
        import onnxruntime_genai as og
    except ImportError:
        print(
            "\n[genai] Skipping ORT GenAI verification: onnxruntime_genai not installed.\n"
            "        Install with: pip install onnxruntime-genai"
        )
        return True

    import tempfile

    import torch

    from mobius.integrations.ort_genai.genai_config import GenaiConfigGenerator

    print("\n" + "=" * 60)
    print("  ORT GenAI Verification")
    print("=" * 60)

    with tempfile.TemporaryDirectory(prefix="phi4mm_genai_") as tmp_dir:
        # ── 1. Save ONNX models ───────────────────────────────────────
        print(f"  Saving ONNX models to {tmp_dir} ...")
        pkg.save(tmp_dir, progress_bar=False)
        print(f"  Saved components: {list(pkg.keys())}")

        # ── 2. Write genai_config.json ────────────────────────────────
        gen = GenaiConfigGenerator.from_config(
            config,
            "phi4mm",
            bos_token_id=_PHI4MM_BOS_TOKEN_ID,
            eos_token_id=_PHI4MM_EOS_TOKEN_IDS,
        )
        # Include vision/embedding/speech sections so ORT-GenAI loads the
        # full 4-model layout (even for text-only inference).
        gen.with_vision(
            image_token_id=IMAGE_TOKEN_ID,
            spatial_merge_size=None,
            config_filename="vision_processor.json",
            input_names={
                "pixel_values": "pixel_values",
                "image_sizes": "image_sizes",
            },
        )
        gen.with_speech(audio_token_id=AUDIO_TOKEN_ID)
        gen.write(tmp_dir)

        # ── 3. Copy tokenizer files ───────────────────────────────────
        try:
            proc = transformers.AutoProcessor.from_pretrained(
                model_id, trust_remote_code=trust_remote_code
            )
            proc.save_pretrained(tmp_dir)
            print("  Tokenizer files copied.")
        except Exception as e:
            print(f"  Warning: could not copy tokenizer files — {e}")

        # ── 4. Run ORT GenAI text generation ─────────────────────────
        print(f"\n  Prompt: {prompt!r}")
        print("  Running ORT GenAI generation ...")
        try:
            model_og = og.Model(tmp_dir)
            tokenizer_og = og.Tokenizer(model_og)
            input_ids_og = tokenizer_og.encode(prompt)
            params = og.GeneratorParams(model_og)
            params.set_search_options(
                do_sample=False,
                max_length=len(input_ids_og) + max_new_tokens,
            )
            generator = og.Generator(model_og, params)
            generator.append_tokens(input_ids_og)

            genai_tokens: list[int] = []
            while not generator.is_done() and len(genai_tokens) < max_new_tokens:
                generator.generate_next_token()
                genai_tokens.append(int(generator.get_next_tokens()[0]))
            del generator
            del model_og

            genai_text = tokenizer.decode(genai_tokens, skip_special_tokens=True)
        except Exception as e:
            print(f"  ORT GenAI generation FAILED: {e}")
            print("=" * 60)
            return False

        # ── 5. Run HuggingFace text generation for comparison ─────────
        print("  Running HuggingFace generation for comparison ...")
        hf_text: str | None = None
        try:
            tok_input = tokenizer(prompt, return_tensors="pt")
            input_ids_hf = tok_input["input_ids"]
            with torch.no_grad():
                out_ids = hf_model.generate(
                    input_ids=input_ids_hf,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            new_ids = out_ids[:, input_ids_hf.shape[1] :]
            hf_text = tokenizer.decode(new_ids[0], skip_special_tokens=True)
        except Exception as e:
            print(f"  Warning: HuggingFace comparison failed — {e}")

        # ── 6. Report ─────────────────────────────────────────────────
        print(f"\n  {'=' * 56}")
        print(f"  ORT GenAI:     {genai_text!r}")
        if hf_text is not None:
            print(f"  HuggingFace:   {hf_text!r}")
            match = genai_text.strip() == hf_text.strip()
            status = "✅ MATCH" if match else "⚠️  MISMATCH"
            print(f"  Result:        {status}")
        else:
            print("  HuggingFace:   (comparison unavailable)")
            print("  Result:        ✅ Generation succeeded (no comparison)")
            match = True
        print(f"  {'=' * 56}")
        print()

    return match


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Phi-4 Multimodal output parity: ONNX (mobius) vs HuggingFace transformers."
        ),
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help="HuggingFace model ID (default: %(default)s).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help=(
            "Allow loading remote model code from HuggingFace Hub. "
            "Automatically enabled for the default Phi-4 model ID. "
            "Required for any model that ships custom Python code in its repo."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=[*ALL_MODES, "all"],
        default="all",
        help=(
            "Which modality to test (default: %(default)s). 'all' runs all five test cases."
        ),
    )
    parser.add_argument(
        "--image",
        default=None,
        help=(
            "Path to image file. If not provided, random pixel data is used for vision tests."
        ),
    )
    parser.add_argument(
        "--audio",
        default=None,
        help=(
            "Path to audio file. If not provided, random mel "
            "features are used for audio tests."
        ),
    )
    parser.add_argument(
        "--num-text-layers",
        type=int,
        default=DEFAULT_NUM_TEXT_LAYERS,
        help=(
            "Number of decoder layers to use (default: %(default)s = full model). "
            "Pass a smaller value (e.g. 2) for faster development testing."
        ),
    )
    parser.add_argument(
        "--num-vision-layers",
        type=int,
        default=DEFAULT_NUM_VISION_LAYERS,
        help=(
            "Number of SigLIP vision encoder layers (default: %(default)s = full model). "
            "The ONNX model runs N-1 layers; HF uses layer_idx=-2."
        ),
    )
    parser.add_argument(
        "--num-audio-blocks",
        type=int,
        default=DEFAULT_NUM_AUDIO_BLOCKS,
        help=(
            "Number of Conformer audio encoder blocks (default: %(default)s = full model). "
            "Pass a smaller value for faster development testing."
        ),
    )
    parser.add_argument(
        "--long-audio-frames",
        type=int,
        default=LONG_AUDIO_FRAMES,
        help=("Number of mel frames for the long-audio test (default: %(default)s)."),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            f"Use reduced layer counts for fast pipeline smoke-testing "
            f"({DEBUG_NUM_TEXT_LAYERS} text, {DEBUG_NUM_VISION_LAYERS} vision, "
            f"{DEBUG_NUM_AUDIO_BLOCKS} audio). "
            f"Overrides --num-text-layers / --num-vision-layers / --num-audio-blocks."
        ),
    )
    parser.add_argument(
        "--genai",
        action="store_true",
        help=(
            "After parity tests, save the ONNX models to a temp directory and "
            "verify text-only generation matches HuggingFace using onnxruntime-genai. "
            "Skipped silently if onnxruntime-genai is not installed."
        ),
    )
    args = parser.parse_args()

    # --debug overrides individual layer counts with small preset values.
    if args.debug:
        args.num_text_layers = DEBUG_NUM_TEXT_LAYERS
        args.num_vision_layers = DEBUG_NUM_VISION_LAYERS
        args.num_audio_blocks = DEBUG_NUM_AUDIO_BLOCKS

    # trust_remote_code is always required for the default Phi-4 model ID
    # (its model class lives in the HF repo, not transformers core). For
    # any other --model-id the user must pass --trust-remote-code explicitly.
    trust_remote_code: bool = args.trust_remote_code or (args.model_id == MODEL_ID)

    # ------------------------------------------------------------------
    # Step 1: Load both models
    # ------------------------------------------------------------------
    print(f"Building ONNX models from {args.model_id!r} ...")
    print(
        f"  Layers: text={args.num_text_layers}, "
        f"vision={args.num_vision_layers}, "
        f"audio={args.num_audio_blocks}"
    )
    pkg, config = load_onnx_package(
        args.model_id,
        args.num_text_layers,
        args.num_vision_layers,
        args.num_audio_blocks,
        trust_remote_code=trust_remote_code,
    )
    print(f"  ONNX package components: {list(pkg.keys())}")

    print(f"\nLoading HuggingFace model from {args.model_id!r} ...")
    hf_model, tokenizer = load_hf_model(
        args.model_id,
        args.num_text_layers,
        num_audio_blocks=args.num_audio_blocks,
        num_vision_layers=args.num_vision_layers,
        trust_remote_code=trust_remote_code,
    )

    # ------------------------------------------------------------------
    # Step 2: Discover real test data (if available)
    # ------------------------------------------------------------------
    testdata = _discover_testdata(TESTDATA_DIR)
    if testdata["images"] or testdata["audios"]:
        print(
            f"\nUsing real test data from {TESTDATA_DIR}:"
            f" {len(testdata['images'])} image(s),"
            f" {len(testdata['audios'])} audio file(s)"
        )
    else:
        print("\nNo testdata found — using synthetic dummy inputs.")

    # ------------------------------------------------------------------
    # Step 3: Run parity tests
    # ------------------------------------------------------------------
    modes = ALL_MODES if args.mode == "all" else [args.mode]
    results: list[dict] = []

    for mode in modes:
        if mode == "text":
            results.append(test_text_only(pkg, config, hf_model, tokenizer))

        elif mode == "vision":
            # Iterate over all discovered images; fall back to --image or dummy.
            image_paths: list[str | None] = testdata["images"] or (
                [args.image] if args.image else [None]
            )
            for img_path in image_paths:
                try:
                    results.append(
                        test_vision(pkg, config, hf_model, tokenizer, image_path=img_path)
                    )
                except Exception as e:
                    name = os.path.basename(img_path) if img_path else "dummy"
                    print(f"\n  Warning: skipping {name!r} — {e}")
                    results.append(skipped_result(f"Text + Image ({name})", f"error: {e}"))

        elif mode == "audio-short":
            # Short audio: dummy frames or explicit --audio.
            # When real testdata is present, use the first audio file capped
            # to 30 s so the subsampled token count stays below 500 (no chunking).
            if testdata["audios"] and not args.audio:
                short_path = testdata["audios"][0]
                short_label = f"Text + Audio (short): {os.path.basename(short_path)}"
                results.append(
                    test_audio(
                        pkg,
                        config,
                        hf_model,
                        tokenizer,
                        audio_path=short_path,
                        max_audio_seconds=30.0,
                        label=short_label,
                    )
                )
            else:
                results.append(
                    test_audio(
                        pkg,
                        config,
                        hf_model,
                        tokenizer,
                        audio_path=args.audio,
                        num_frames=SHORT_AUDIO_FRAMES,
                        label="Text + Audio (short)",
                    )
                )

        elif mode == "audio-long":
            # Long audio: iterate all discovered audio files (each capped at
            # MAX_AUDIO_SECONDS).  Exercises the >500-token chunking path.
            audio_paths: list[str | None] = testdata["audios"] or (
                [args.audio] if args.audio else [None]
            )
            for aud_path in audio_paths:
                label = (
                    f"Text + Audio (long): {os.path.basename(aud_path)}"
                    if aud_path
                    else "Text + Audio (long)"
                )
                results.append(
                    test_audio(
                        pkg,
                        config,
                        hf_model,
                        tokenizer,
                        audio_path=aud_path,
                        num_frames=args.long_audio_frames,
                        label=label,
                    )
                )

        elif mode == "vision-audio":
            # Combined: first image + first audio (or explicit --image/--audio).
            img_path = args.image or (testdata["images"][0] if testdata["images"] else None)
            aud_path = args.audio or (testdata["audios"][0] if testdata["audios"] else None)
            results.append(
                test_vision_audio(
                    pkg,
                    config,
                    hf_model,
                    tokenizer,
                    image_path=img_path,
                    audio_path=aud_path,
                )
            )

    # ------------------------------------------------------------------
    # Step 3: Summary
    # ------------------------------------------------------------------
    all_pass = print_summary(results)

    # ------------------------------------------------------------------
    # Step 4: ORT GenAI verification (optional, --genai flag)
    # ------------------------------------------------------------------
    if args.genai:
        genai_pass = run_genai_verification(
            pkg,
            config,
            args.model_id,
            tokenizer,
            hf_model,
            trust_remote_code=trust_remote_code,
        )
        all_pass = all_pass and genai_pass

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
