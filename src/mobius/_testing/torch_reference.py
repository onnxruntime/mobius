# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""PyTorch/HuggingFace reference model helpers for integration testing."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _mage_vl_optional_streammind_import(model_id: str):
    """Treat StreamMind's mamba-ssm dependency as optional for base Mage-VL.

    The remote ``modeling_mage_vl.py`` imports ``streammind_gate`` only inside
    StreamMind-specific methods, but Transformers recursively validates that
    sibling module and otherwise requires mamba-ssm even for ordinary
    image/video generation. mamba-ssm has no Windows wheel and is not used by
    the base checkpoint path exercised here.
    """
    if model_id.lower() != "microsoft/mage-vl":
        yield
        return

    import transformers.dynamic_module_utils as dynamic_module_utils

    original_get_imports = dynamic_module_utils.get_imports

    def _get_imports(filename):
        imports = original_get_imports(filename)
        if Path(filename).name == "streammind_gate.py":
            return [name for name in imports if name != "mamba_ssm"]
        return imports

    dynamic_module_utils.get_imports = _get_imports
    try:
        yield
    finally:
        dynamic_module_utils.get_imports = original_get_imports


def _load_mage_compatible(model_id: str, loader, *args, **kwargs):
    with _mage_vl_optional_streammind_import(model_id):
        return loader(*args, **kwargs)


def _install_dynamic_cache_legacy_shims() -> None:
    """Restore ``DynamicCache`` methods removed in transformers 5.x.

    transformers 5.x removed ``DynamicCache.from_legacy_cache`` and
    ``DynamicCache.get_usable_length``, but some ``trust_remote_code``
    checkpoints (e.g. Phi-3.5 family) still call them from their bundled
    modeling code. Re-add minimal, behaviour-preserving implementations so
    reference generation continues to work on transformers 5.x. Both the
    causal-LM and multimodal loaders rely on this shim.
    """
    import transformers

    if not hasattr(transformers.DynamicCache, "from_legacy_cache"):

        @classmethod  # type: ignore[misc]
        def _from_legacy_cache(cls, past_key_values=None):  # type: ignore[misc]
            cache = cls()
            if past_key_values is not None:
                for i, (k, v) in enumerate(past_key_values):
                    cache.update(k, v, i)
            return cache

        transformers.DynamicCache.from_legacy_cache = _from_legacy_cache

    if not hasattr(transformers.DynamicCache, "to_legacy_cache"):

        def _to_legacy_cache(self):  # type: ignore[misc]
            legacy_cache = []
            for layer in getattr(self, "layers", []):
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
                if keys is None or values is None:
                    continue
                legacy_cache.append((keys, values))
            return tuple(legacy_cache)

        transformers.DynamicCache.to_legacy_cache = _to_legacy_cache  # type: ignore[method-assign]

    if not hasattr(transformers.DynamicCache, "seen_tokens"):

        def _seen_tokens(self):  # type: ignore[misc]
            # Legacy alias for the number of tokens the cache has seen so
            # far, which for a full (non-windowed) cache equals the current
            # sequence length of layer 0.
            return self.get_seq_length(0)

        transformers.DynamicCache.seen_tokens = property(_seen_tokens)  # type: ignore[assignment]

    if not hasattr(transformers.DynamicCache, "get_usable_length"):

        def _get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:  # type: ignore[misc]
            # Mirrors the removed transformers implementation: the usable
            # length is the number of tokens ALREADY cached for this layer
            # (the past length), NOT including the incoming ``new_seq_length``.
            # Returning ``previous + new`` double-counts the query tokens and
            # makes callers (e.g. the Phi-3.5 bundled modeling code) build a
            # causal mask of the wrong size.
            return self.get_seq_length(layer_idx)

        transformers.DynamicCache.get_usable_length = _get_usable_length  # type: ignore[method-assign]


def _fix_nemotron_h_init_weights(
    model: torch.nn.Module, model_id: str, revision: str | None = None
) -> None:
    """Restore Mamba2 params from checkpoint after HF clobbers them.

    The NemotronH remote-code ``_init_weights`` re-initialises several
    parameters *after* ``from_pretrained`` loads the checkpoint:

    - ``dt_bias``: overwritten with ``torch.rand(...)``
    - ``out_proj.weight`` (in Mamba mixer layers): overwritten with
      ``kaiming_uniform_`` then scaled by ``1/sqrt(n_layers)`` when
      ``rescale_prenorm_residual`` is True

    This helper reads the original values back from the safetensors
    files on disk and patches them in-place.
    """
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    if model_type != "nemotron_h":
        return

    try:
        from safetensors import safe_open
    except ImportError:
        return

    import glob
    import os

    from huggingface_hub import snapshot_download

    # Resolve the exact snapshot directory used by HF for this model,
    # avoiding lexicographic guessing across multiple cached revisions.
    try:
        snapshot = snapshot_download(model_id, revision=revision, local_files_only=True)
    except Exception:
        logger.warning(
            "NemotronH init_weights fix: could not resolve snapshot for %s",
            model_id,
        )
        return

    safetensor_files = sorted(glob.glob(os.path.join(snapshot, "*.safetensors")))

    # Collect parameter names that _init_weights corrupts:
    # 1. All dt_bias params (Mamba2 layers)
    # 2. mixer.out_proj.weight params (rescale_prenorm_residual)
    corrupted_suffixes = {"dt_bias"}
    if getattr(config, "rescale_prenorm_residual", False):
        corrupted_suffixes.add("mixer.out_proj.weight")

    patched = 0
    state = model.state_dict()
    for f in safetensor_files:
        with safe_open(f, framework="pt") as st:
            # safe_open objects aren't directly iterable
            keys = st.keys()
            for key in keys:
                if not any(key.endswith(s) for s in corrupted_suffixes):
                    continue
                if key not in state:
                    continue
                ckpt_val = st.get_tensor(key)
                param = state[key]
                with torch.no_grad():
                    param.copy_(ckpt_val.to(param.device, dtype=param.dtype))
                patched += 1

    # Write the fixed values back into the live model
    if patched:
        model.load_state_dict(state, strict=False)
        logger.info(
            "NemotronH: restored %d params from checkpoint (dt_bias + out_proj.weight)",
            patched,
        )
    else:
        logger.warning(
            "NemotronH init_weights fix: no corrupted params found in checkpoint for %s",
            model_id,
        )


def load_torch_model(
    model_id: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    trust_remote_code: bool = True,
    revision: str | None = None,
):
    """Load a HuggingFace causal LM model for reference inference.

    Args:
        model_id: HuggingFace model identifier.
        dtype: Model dtype (default float32 for numerical comparison).
        device: Device to load on.
        trust_remote_code: Whether to load the checkpoint's bundled modeling
            code. Defaults to ``True`` for backward compatibility. Pass
            ``False`` for models (e.g. Phi-3.5-mini) whose ``model_type`` is
            natively supported by the installed transformers, so the
            transformers-5.x-compatible implementation is used instead of an
            older bundled ``modeling_*.py`` that relies on removed cache APIs.
        revision: Immutable HuggingFace revision used for the tokenizer,
            config, weights, and any Nemotron-H weight repair.

    Returns:
        Tuple of (model, tokenizer).
    """
    import transformers

    _install_dynamic_cache_legacy_shims()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=trust_remote_code
    )

    # NemotronH: disable rescale_prenorm_residual before loading to
    # prevent _init_weights from corrupting out_proj.weight with
    # random kaiming_uniform_ initialization after checkpoint loading.
    config = transformers.AutoConfig.from_pretrained(
        model_id, revision=revision, trust_remote_code=trust_remote_code
    )
    if getattr(config, "model_type", None) == "nemotron_h":
        config.rescale_prenorm_residual = False

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        dtype=dtype,
        device_map=device,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    _fix_nemotron_h_init_weights(model, model_id, revision)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_torch_multimodal_model(
    model_id: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    revision: str | None = None,
):
    """Load a HuggingFace multimodal model for reference inference.

    Uses AutoModelForImageTextToText for vision-language models.

    Args:
        model_id: HuggingFace model identifier.
        dtype: Model dtype (default float32 for numerical comparison).
        device: Device to load on.
        revision: Optional immutable HuggingFace revision used for tokenizer,
            processor, config, and weight loading.

    Returns:
        Tuple of (model, tokenizer, image_processor).
    """
    import transformers

    hub_kwargs: dict[str, object] = {"trust_remote_code": True}
    if revision is not None:
        hub_kwargs["revision"] = revision

    tokenizer = _load_mage_compatible(
        model_id,
        transformers.AutoTokenizer.from_pretrained,
        model_id,
        **hub_kwargs,
    )
    processor = _load_mage_compatible(
        model_id,
        transformers.AutoProcessor.from_pretrained,
        model_id,
        **hub_kwargs,
    )

    # Shim: transformers 5.x removed DynamicCache.from_legacy_cache and
    # DynamicCache.get_usable_length, but some trust_remote_code models
    # (e.g. Phi-3.5-vision-instruct) still call them.
    _install_dynamic_cache_legacy_shims()

    # Load config and force eager attention so flash_attn is not required.
    # Some models (e.g. Phi-3.5-vision-instruct) hardcode flash_attention_2
    # in their config.json, which causes an ImportError when flash_attn is
    # not installed.
    config = _load_mage_compatible(
        model_id,
        transformers.AutoConfig.from_pretrained,
        model_id,
        **hub_kwargs,
    )
    config._attn_implementation = "eager"

    # Some trust_remote_code VLMs (e.g. Phi-3-Vision) are registered as
    # AutoModelForCausalLM, not AutoModelForImageTextToText.  Fall back
    # gracefully so golden generation works for both.
    #
    # The weight-dtype keyword was renamed from ``torch_dtype`` to ``dtype`` in
    # transformers 5.x; support both so offline golden generation also works on
    # the older transformers (e.g. 4.43) required by 4.x-era remote-code models.
    base_kwargs = dict(config=config, device_map=device, **hub_kwargs)

    def _load_from_pretrained(auto_cls):
        try:
            return _load_mage_compatible(
                model_id,
                auto_cls.from_pretrained,
                model_id,
                dtype=dtype,
                **base_kwargs,
            )
        except TypeError:
            return _load_mage_compatible(
                model_id,
                auto_cls.from_pretrained,
                model_id,
                torch_dtype=dtype,
                **base_kwargs,
            )

    try:
        image_text_to_text_cls = transformers.AutoModelForImageTextToText
    except AttributeError:
        # Older transformers (e.g. 4.43, used for offline golden generation of
        # 4.x-era remote-code checkpoints) predate AutoModelForImageTextToText.
        image_text_to_text_cls = None
    model = None
    if image_text_to_text_cls is not None:
        try:
            model = _load_from_pretrained(image_text_to_text_cls)
        except (ValueError, KeyError):
            model = None
    if model is None:
        model = _load_from_pretrained(transformers.AutoModelForCausalLM)

    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, processor


@torch.no_grad()
def torch_forward(
    model,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    position_ids: np.ndarray,
    past_key_values: object | None = None,
) -> tuple[np.ndarray, object]:
    """Run a single forward pass on a HuggingFace causal LM model.

    Args:
        model: HuggingFace model in eval mode.
        input_ids: [batch, seq_len] int64 numpy array.
        attention_mask: [batch, total_seq_len] int64 numpy array.
        position_ids: [batch, seq_len] int64 numpy array.
        past_key_values: Optional list of (key, value) numpy array tuples, or
            an opaque HuggingFace Cache for hybrid recurrent models.

    Returns:
        Tuple of logits and either a list of KV numpy tuples or an opaque
        HuggingFace Cache when model-specific recurrent state must be retained.
    """
    import inspect

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    ids_t = torch.from_numpy(input_ids).to(device)
    mask_t = torch.from_numpy(attention_mask).to(device)
    pos_t = torch.from_numpy(position_ids).to(device)

    kwargs: dict = {
        "input_ids": ids_t,
        "attention_mask": mask_t,
        "use_cache": True,
    }

    # Some models (Falcon, Mamba) don't accept position_ids
    fwd_sig = inspect.signature(model.forward)
    if "position_ids" in fwd_sig.parameters:
        kwargs["position_ids"] = pos_t

    if past_key_values is not None:
        from transformers.cache_utils import Cache, DynamicCache

        if isinstance(past_key_values, Cache):
            # Hybrid caches carry model-specific recurrent state (for example
            # LFM2 conv windows) that cannot be reconstructed from KV pairs.
            kwargs["past_key_values"] = past_key_values
        else:
            cache = DynamicCache()
            for layer_idx, (k, v) in enumerate(past_key_values):
                cache.update(
                    torch.from_numpy(k).to(device=device, dtype=dtype),
                    torch.from_numpy(v).to(device=device, dtype=dtype),
                    layer_idx,
                )
            kwargs["past_key_values"] = cache

    try:
        outputs = model(**kwargs)
    except ValueError as e:
        # All-attention hybrid models (e.g. GraniteMoeHybrid granite-4.0-1b)
        # trip transformers' recurrent-mask builder, which assumes the hybrid
        # cache contains a linear-attention (Mamba) layer: "`has_previous_state`
        # can only be called on LinearAttention layers". A single-pass forward's
        # logits don't depend on caching, so retry without a cache.
        if "has_previous_state" not in str(e) or "past_key_values" in kwargs:
            raise
        kwargs["use_cache"] = False
        outputs = model(**kwargs)
    logits = outputs.logits.cpu().numpy()

    # Extract KV cache if available (Mamba models don't have it)
    present_kv: list[tuple[np.ndarray, np.ndarray]] = []
    cache = getattr(outputs, "past_key_values", None)
    layer_types = getattr(model.config, "layer_types", None) or []
    if "conv" in layer_types:
        return logits, cache

    if cache is not None and hasattr(cache, "layers"):
        for layer_idx in range(len(cache.layers)):
            layer_cache = cache.layers[layer_idx]
            # Mamba layers use LinearAttentionLayer without keys/values
            if not hasattr(layer_cache, "keys"):
                continue
            k = layer_cache.keys.cpu().numpy()
            v = layer_cache.values.cpu().numpy()
            present_kv.append((k, v))

    return logits, present_kv


@torch.no_grad()
def torch_multimodal_forward(
    model,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    position_ids: np.ndarray,
    pixel_values: np.ndarray,
    past_key_values: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Run a single forward pass on a HuggingFace multimodal model.

    Args:
        model: HuggingFace multimodal model in eval mode.
        input_ids: [batch, seq_len] int64 numpy array.
        attention_mask: [batch, total_seq_len] int64 numpy array.
        position_ids: [batch, seq_len] int64 numpy array.
        pixel_values: [batch, channels, height, width] float32 numpy array.
        past_key_values: Optional list of (key, value) numpy array tuples.

    Returns:
        Tuple of (logits as numpy, list of (key, value) numpy tuples).
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    ids_t = torch.from_numpy(input_ids).to(device)
    mask_t = torch.from_numpy(attention_mask).to(device)
    pos_t = torch.from_numpy(position_ids).to(device)
    pv_t = torch.from_numpy(pixel_values).to(device=device, dtype=dtype)

    kwargs: dict = {
        "input_ids": ids_t,
        "attention_mask": mask_t,
        "position_ids": pos_t,
        "pixel_values": pv_t,
        "use_cache": True,
    }

    if past_key_values is not None:
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(past_key_values):
            cache.update(
                torch.from_numpy(k).to(device=device, dtype=dtype),
                torch.from_numpy(v).to(device=device, dtype=dtype),
                layer_idx,
            )
        kwargs["past_key_values"] = cache

    outputs = model(**kwargs)
    logits = outputs.logits.cpu().numpy()

    present_kv = []
    cache = outputs.past_key_values
    for layer_idx in range(len(cache.layers)):
        k = cache.layers[layer_idx].keys.cpu().numpy()
        v = cache.layers[layer_idx].values.cpu().numpy()
        present_kv.append((k, v))

    return logits, present_kv


# ---------------------------------------------------------------------------
# Encoder-only models (BERT, RoBERTa, DistilBERT, etc.)
# ---------------------------------------------------------------------------


def load_torch_encoder_model(
    model_id: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
):
    """Load a HuggingFace encoder-only model for reference inference.

    Uses AutoModel (not AutoModelForCausalLM) for encoder-only architectures.

    Returns:
        Tuple of (model, tokenizer).
    """
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = transformers.AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


@torch.no_grad()
def torch_encoder_forward(
    model,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    token_type_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Run a single forward pass on a HuggingFace encoder-only model.

    Returns:
        last_hidden_state as numpy array [batch, seq_len, hidden_size].
    """
    device = next(model.parameters()).device
    kwargs: dict = {
        "input_ids": torch.from_numpy(input_ids).to(device),
        "attention_mask": torch.from_numpy(attention_mask).to(device),
    }
    if token_type_ids is not None:
        kwargs["token_type_ids"] = torch.from_numpy(token_type_ids).to(device)
    outputs = model(**kwargs)
    return outputs.last_hidden_state.cpu().numpy()


# ---------------------------------------------------------------------------
# Seq2seq models (BART, T5, mBART, etc.)
# ---------------------------------------------------------------------------


def load_torch_seq2seq_model(
    model_id: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
):
    """Load a HuggingFace seq2seq model for reference inference.

    Returns:
        Tuple of (model, tokenizer).
    """
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


@torch.no_grad()
def torch_seq2seq_encoder_forward(
    model,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
) -> np.ndarray:
    """Run the encoder of a seq2seq model and return hidden states.

    Returns:
        last_hidden_state as numpy array [batch, seq_len, d_model].
    """
    device = next(model.parameters()).device
    ids_t = torch.from_numpy(input_ids).to(device)
    mask_t = torch.from_numpy(attention_mask).to(device)
    encoder_out = model.get_encoder()(input_ids=ids_t, attention_mask=mask_t)
    return encoder_out.last_hidden_state.cpu().numpy()


@torch.no_grad()
def torch_seq2seq_decoder_forward(
    model,
    decoder_input_ids: np.ndarray,
    encoder_hidden_states: np.ndarray,
    encoder_attention_mask: np.ndarray | None = None,
    past_key_values=None,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Run the decoder of a seq2seq model and return logits + KV cache.

    Returns:
        Tuple of (logits as numpy, past_key_values for next step).
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    kwargs: dict = {
        "decoder_input_ids": torch.from_numpy(decoder_input_ids).to(device),
        "encoder_outputs": (
            torch.from_numpy(encoder_hidden_states).to(device=device, dtype=dtype),
        ),
        "use_cache": True,
    }
    if encoder_attention_mask is not None:
        kwargs["attention_mask"] = torch.from_numpy(encoder_attention_mask).to(device)
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values

    outputs = model(**kwargs)
    logits = outputs.logits.cpu().numpy()
    return logits, outputs.past_key_values


# ---------------------------------------------------------------------------
# Vision models (ViT, DeiT, CLIP Vision, etc.)
# ---------------------------------------------------------------------------


def load_torch_vision_model(
    model_id: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    trust_remote_code: bool = False,
):
    """Load a HuggingFace vision model for reference inference.

    For multi-modal models like CLIP, extracts just the vision sub-model
    so that ``torch_vision_forward`` can run with only ``pixel_values``.

    Returns:
        Tuple of (model, processor).
    """
    import transformers

    processor = transformers.AutoImageProcessor.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )

    # Some vision models (DepthAnything, Segformer) aren't loadable via
    # AutoModel.  Try progressively more specific Auto classes.
    model = None
    auto_classes = [
        transformers.AutoModel,
        transformers.AutoModelForImageClassification,
    ]
    if hasattr(transformers, "AutoModelForDepthEstimation"):
        auto_classes.append(transformers.AutoModelForDepthEstimation)
    if hasattr(transformers, "AutoModelForSemanticSegmentation"):
        auto_classes.append(transformers.AutoModelForSemanticSegmentation)
    if hasattr(transformers, "AutoModelForImageToImage"):
        auto_classes.append(transformers.AutoModelForImageToImage)
    if hasattr(transformers, "AutoModelForObjectDetection"):
        auto_classes.append(transformers.AutoModelForObjectDetection)
    for auto_cls in auto_classes:
        try:
            model = auto_cls.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=trust_remote_code,
            )
            break
        except (ValueError, TypeError):
            continue
    if model is None:
        raise ValueError(f"Could not load {model_id} with any AutoModel variant")
    model.eval()

    # Multi-modal models (CLIP, SigLIP) wrap a vision sub-model that
    # can be called with pixel_values alone.
    if hasattr(model, "vision_model"):
        model = model.vision_model

    return model, processor


@torch.no_grad()
def torch_vision_forward(
    model,
    pixel_values: np.ndarray,
) -> np.ndarray:
    """Run a single forward pass on a HuggingFace vision model.

    Returns:
        Feature tensor as numpy array.  Shape varies by model:
        [B, seq_len, hidden] for ViT-like, [B, C, H, W] for CNN-like,
        or [B, num_classes] for classification heads.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    pv = torch.from_numpy(pixel_values).to(device=device, dtype=dtype)
    outputs = model(pixel_values=pv)
    # Prefer last_hidden_state; fall back to logits or first tensor output.
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state.cpu().numpy()
    if hasattr(outputs, "logits") and outputs.logits is not None:
        return outputs.logits.cpu().numpy()
    # Some models (DepthAnything) return predicted_depth or similar.
    if hasattr(outputs, "predicted_depth") and outputs.predicted_depth is not None:
        return outputs.predicted_depth.cpu().numpy()
    # Generic fallback: accept dict-like ModelOutput, tuple, or list returns.
    if hasattr(outputs, "cpu"):
        return outputs.cpu().numpy()
    values = None
    if hasattr(outputs, "values"):
        values = outputs.values()
    elif isinstance(outputs, (tuple, list)):
        values = outputs
    if values is not None:
        for v in values:
            if hasattr(v, "cpu"):
                return v.cpu().numpy()
    raise ValueError(f"No usable tensor in model outputs: {type(outputs)}")


# ---------------------------------------------------------------------------
# Audio models (Wav2Vec2, HuBERT, WavLM, etc.)
# ---------------------------------------------------------------------------


def load_torch_audio_model(
    model_id: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    trust_remote_code: bool = False,
):
    """Load a HuggingFace audio model for reference inference.

    Args:
        trust_remote_code: Whether to allow executing remote code from the
            model repository.  Defaults to False for safety.

    Returns:
        Tuple of (model, processor).
    """
    import transformers

    # Some audio models (e.g. HuBERT) only have a feature extractor,
    # not a full processor with a tokenizer.  Fall back gracefully.
    try:
        processor = transformers.AutoProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
    except (TypeError, OSError):
        processor = transformers.AutoFeatureExtractor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
    model = transformers.AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    return model, processor


@torch.no_grad()
def torch_audio_forward(
    model,
    input_values: np.ndarray,
) -> np.ndarray:
    """Run a single forward pass on a HuggingFace audio model.

    Returns:
        last_hidden_state as numpy array [batch, seq_len, hidden_size].
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    iv = torch.from_numpy(input_values).to(device=device, dtype=dtype)
    outputs = model(input_values=iv)
    return outputs.last_hidden_state.cpu().numpy()


# ---------------------------------------------------------------------------
# Whisper encoder-decoder models
# ---------------------------------------------------------------------------


@torch.no_grad()
def load_torch_whisper_model(
    model_id: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
):
    """Load a HuggingFace Whisper model for reference inference.

    Returns:
        Tuple of (model, processor).
    """
    import transformers

    processor = transformers.AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    return model, processor


@torch.no_grad()
def torch_whisper_encoder_forward(
    model,
    input_features: np.ndarray,
) -> np.ndarray:
    """Run the Whisper encoder and return encoder hidden states.

    Args:
        model: HuggingFace WhisperForConditionalGeneration in eval mode.
        input_features: [batch, num_mel_bins, audio_seq_len] float32 numpy array.

    Returns:
        encoder_hidden_states as numpy array [batch, seq/2, d_model].
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    feats = torch.from_numpy(input_features).to(device=device, dtype=dtype)
    encoder_out = model.model.encoder(feats)
    return encoder_out.last_hidden_state.cpu().numpy()


@torch.no_grad()
def torch_whisper_decoder_forward(
    model,
    decoder_input_ids: np.ndarray,
    encoder_hidden_states: np.ndarray,
    attention_mask: np.ndarray | None = None,
    past_key_values: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Run the Whisper decoder and return logits + KV cache.

    Args:
        model: HuggingFace WhisperForConditionalGeneration in eval mode.
        decoder_input_ids: [batch, seq_len] int64 numpy array.
        encoder_hidden_states: [batch, enc_seq, d_model] float32 numpy array.
        attention_mask: Optional [batch, total_seq_len] int64.
        past_key_values: Optional list of (key, value) numpy tuples.

    Returns:
        Tuple of (logits as numpy, list of (key, value) numpy tuples).
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    ids_t = torch.from_numpy(decoder_input_ids).to(device)
    enc_t = torch.from_numpy(encoder_hidden_states).to(device=device, dtype=dtype)

    kwargs: dict = {
        "input_ids": ids_t,
        "encoder_hidden_states": enc_t,
        "use_cache": True,
    }

    if attention_mask is not None:
        kwargs["attention_mask"] = torch.from_numpy(attention_mask).to(device)

    if past_key_values is not None:
        from transformers.cache_utils import DynamicCache, EncoderDecoderCache

        self_cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(past_key_values):
            self_cache.update(
                torch.from_numpy(k).to(device=device, dtype=dtype),
                torch.from_numpy(v).to(device=device, dtype=dtype),
                layer_idx,
            )
        # Whisper decoder expects EncoderDecoderCache wrapping self + cross caches
        cross_cache = DynamicCache()
        kwargs["past_key_values"] = EncoderDecoderCache(self_cache, cross_cache)

    outputs = model.model.decoder(**kwargs)
    hidden_states = outputs.last_hidden_state

    # Project to vocab
    logits = model.proj_out(hidden_states).cpu().numpy()

    # Extract self-attention KV cache (Whisper decoder uses EncoderDecoderCache)
    present_kv = []
    cache = outputs.past_key_values
    # EncoderDecoderCache wraps self_attention_cache + cross_attention_cache
    self_cache = getattr(cache, "self_attention_cache", cache)
    for layer_idx in range(len(self_cache.layers)):
        k = self_cache.layers[layer_idx].keys.cpu().numpy()
        v = self_cache.layers[layer_idx].values.cpu().numpy()
        present_kv.append((k, v))

    return logits, present_kv
