# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Weight-quantization configuration parsed from HuggingFace configs."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class QuantizationConfig:
    """Weight quantization parameters parsed from HuggingFace configs.

    Captures the settings from ``quantization_config`` in HuggingFace model
    configs (GPTQ, AWQ, etc.) so models can decide whether to use
    :class:`~mobius.components.QuantizedLinear` instead of
    :class:`~mobius.components.Linear`.
    """

    bits: int = 4
    group_size: int = 128
    quant_method: str = "none"
    sym: bool = True
    # When True, zero_points is a per-block float tensor rather than a
    # bit-packed uint8. Required for codebooks with non-integer offsets
    # (e.g. Tencent SEQ uses 1.5).
    float_zero_point: bool = False
    # When True, the input embedding table is block-wise quantized and is
    # looked up with GatherBlockQuantized instead of a plain Gather. Used by
    # Olive RTN exports and quantized GGUF imports.
    quantize_embeddings: bool = False
    # When True, the LM head projection is block-wise quantized (MatMulNBits).
    # Used by Olive RTN exports and tied quantized GGUF imports.
    quantize_lm_head: bool = False
    # When True, the input embedding and LM head share one weight table. Olive
    # RTN records this in its own config (``tie_word_embeddings``) and may clear
    # the model's top-level flag, so it is tracked here independently.
    tie_word_embeddings: bool = False

    @classmethod
    def from_transformers(cls, hf_config) -> QuantizationConfig | None:
        """Parse ``quantization_config`` from a HuggingFace config.

        Returns ``None`` when no quantization is configured.
        """
        qc = getattr(hf_config, "quantization_config", None)
        if qc is None:
            return None
        # qc can be a dict or a HF QuantizationConfig object
        if hasattr(qc, "to_dict"):
            qc = qc.to_dict()
        if not isinstance(qc, dict):
            return None
        method = qc.get("quant_method", "none")
        if method == "none":
            return None
        # FP8 per-tensor quantization (float8_e4m3fn + scalar scale)
        # is handled by dtype casting in _assign_weight(), not by
        # QuantizedLinear block quantization.
        if method == "fp8":
            return None
        return cls(
            bits=qc.get("bits", 4),
            group_size=qc.get("group_size", 128),
            quant_method=method,
            sym=qc.get("sym", qc.get("symmetric", True)),
            quantize_embeddings=bool(qc.get("embeds", False)),
            quantize_lm_head=bool(qc.get("lm_head", False)),
            tie_word_embeddings=bool(qc.get("tie_word_embeddings", False)),
        )
