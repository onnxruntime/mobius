# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generate EP-specific genai_config.json sections for onnxruntime-genai.

The ``genai_config.json`` file controls how ORT GenAI loads and runs a model.
Different execution providers need different session options, provider options,
and — for TRT-RTX — sliding window configuration and modified KV cache dim
naming.

Reference: onnxruntime-genai ``base.py`` lines 542-662 and 673-688.
"""

from __future__ import annotations

from typing import Any

from mobius._execution_providers import ep_registry

# ORT GenAI provider name mapping (internal name → ORT GenAI provider string).
# GenAI expects short lowercase names (e.g. "cuda", "dml"), not the full
# ORT EP class names (e.g. "CUDAExecutionProvider").
_ORT_PROVIDER_NAMES: dict[str, str] = {
    "cpu": "cpu",
    "cuda": "cuda",
    "dml": "dml",
    "webgpu": "webgpu",
    "trt-rtx": "NvTensorRtRtx",
    "qnn": "QNN",
}


def make_provider_options(
    ep: str,
) -> list[dict[str, dict[str, str]]]:
    """Build the ``provider_options`` list for genai_config.json.

    Graph capture is driven entirely by the EP's registered
    ``EpCapabilities.enable_graph_capture`` flag (the single source of truth).

    Args:
        ep: Execution provider name (``"cpu"``, ``"cuda"``, ``"dml"``,
            ``"webgpu"``, ``"trt-rtx"``).

    Returns:
        A list with a single dict mapping the EP name to its options.
        Empty list for CPU (no provider_options needed).
    """
    if ep == "cpu":
        return []

    ep_name = _ORT_PROVIDER_NAMES.get(ep, ep)
    # Start from EP's registered defaults; fall back to empty if EP unknown.
    caps = ep_registry.get(ep)
    options = dict(caps.provider_options) if caps else {}

    # Graph capture comes from the EP's registered capability flag (the registry
    # is the single source of truth). Translate it into the EP-specific option.
    graph_capture = bool(caps and caps.enable_graph_capture)

    if ep == "webgpu":
        options["enableGraphCapture"] = "1" if graph_capture else "0"
        options["validationMode"] = "disabled" if graph_capture else "basic"
    elif ep in ("cuda", "trt-rtx"):
        # CUDA and TRT-RTX (NvTensorRtRtx) use the CUDA-graph option key.
        options["enable_cuda_graph"] = "1" if graph_capture else "0"

    return [{ep_name: options}]


def make_sliding_window_config(
    *,
    window_size: int,
    num_layers: int,
    is_local_fn: Any | None = None,
) -> dict[str, Any] | None:
    """Build the ``sliding_window`` config dict for TRT-RTX.

    Args:
        window_size: The sliding window size. If ``<= 0``, returns ``None``.
        num_layers: Total number of decoder layers.
        is_local_fn: Optional callable ``(layer_id: int) -> bool`` that
            returns ``True`` for layers using sliding window attention.
            When ``None``, all layers are assumed to use sliding window.

    Returns:
        A dict suitable for ``genai_config["model"]["decoder"]["sliding_window"]``,
        or ``None`` if sliding window is not active.
    """
    if window_size <= 0:
        return None

    if is_local_fn is not None:
        layers = [i for i in range(num_layers) if is_local_fn(i)]
    else:
        layers = list(range(num_layers))

    return {
        "window_size": window_size,
        "slide_key_value_cache": False,
        "slide_inputs": False,
        "layers": layers,
    }


def make_kv_cache_dim_name(
    dim_name: str,
    *,
    ep: str,
    is_sliding_layer: bool = False,
) -> str:
    """Return the correct KV cache sequence dimension name.

    For TRT-RTX with sliding window layers, ``"sequence"`` in the dimension
    name is replaced with ``"sliding"`` — e.g. ``"past_sequence_length"``
    becomes ``"past_sliding_length"`` — so that the runtime allocates
    fixed-size sliding window buffers instead of unbounded KV caches.

    Args:
        dim_name: The base dimension name (e.g. ``"past_sequence_length"``).
        ep: Target execution provider.
        is_sliding_layer: Whether this layer uses sliding window attention.

    Returns:
        The (possibly modified) dimension name.
    """
    if ep == "trt-rtx" and is_sliding_layer:
        return dim_name.replace("sequence", "sliding")
    return dim_name


def make_genai_decoder_config(
    ep: str,
    *,
    filename: str = "model.onnx",
    head_size: int,
    hidden_size: int,
    num_attention_heads: int,
    num_hidden_layers: int,
    num_key_value_heads: int,
    sliding_window_size: int = 0,
    is_local_fn: Any | None = None,
) -> dict[str, Any]:
    """Build the ``model.decoder`` section of genai_config.json.

    This assembles session_options, provider_options, I/O name templates,
    model dimensions, and optional sliding window config into the dict
    structure expected by ORT GenAI.

    Args:
        ep: Target execution provider.
        filename: ONNX model filename.
        head_size: Attention head dimension.
        hidden_size: Model hidden size.
        num_attention_heads: Number of attention heads.
        num_hidden_layers: Number of decoder layers.
        num_key_value_heads: Number of KV heads.
        sliding_window_size: Sliding window size for TRT-RTX (0 = disabled).
        is_local_fn: Per-layer sliding window predicate for TRT-RTX.

    Returns:
        Dict for ``genai_config["model"]["decoder"]``.
    """
    provider_options = make_provider_options(ep)

    decoder: dict[str, Any] = {
        "session_options": {
            "log_id": "onnxruntime-genai",
            "provider_options": provider_options,
        },
        "filename": filename,
        "head_size": head_size,
        "hidden_size": hidden_size,
        "inputs": {
            "input_ids": "input_ids",
            "attention_mask": "attention_mask",
            "position_ids": "position_ids",
            "past_key_names": "past_key_values.%d.key",
            "past_value_names": "past_key_values.%d.value",
        },
        "outputs": {
            "logits": "logits",
            "present_key_names": "present.%d.key",
            "present_value_names": "present.%d.value",
        },
        "num_attention_heads": num_attention_heads,
        "num_hidden_layers": num_hidden_layers,
        "num_key_value_heads": num_key_value_heads,
    }

    # TRT-RTX sliding window config
    if ep == "trt-rtx":
        sw_config = make_sliding_window_config(
            window_size=sliding_window_size,
            num_layers=num_hidden_layers,
            is_local_fn=is_local_fn,
        )
        if sw_config is not None:
            decoder["sliding_window"] = sw_config

    return decoder
