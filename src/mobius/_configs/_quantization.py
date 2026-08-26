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
    # Used by Olive RTN exports and quantized GGUF imports.
    quantize_lm_head: bool = False
    # When True, multimodal vision projections are block-wise quantized.
    # Olive RTN records this when ``quantize_vision`` is enabled.
    quantize_vision: bool = False
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
        # NVIDIA ModelOpt NVFP4/FP8 checkpoints (e.g. quantized Qwen3.6) encode
        # weights as packed E2M1 (fp4) / float8 with block + global scales — a
        # layout the INT4 ``QuantizedLinear``/``MatMulNBits`` path below would
        # silently mis-dequantize. Checked before the ``quant_method == "none"``
        # early-return because ModelOpt may name its scheme only via
        # ``quant_algo``/``quant_cfg`` (no ``quant_method``). Fail loudly: the
        # reconstruction math is available in :mod:`mobius.integrations.modelopt`,
        # but the full weight-load integration + native routed-expert NVFP4 QMoE
        # emission (CUDA/Blackwell, ``onnxruntime_USE_FP4_QMOE=ON``) are not wired.
        from mobius.integrations.modelopt import is_modelopt_quant_config

        if is_modelopt_quant_config(qc):
            raise NotImplementedError(
                "NVIDIA ModelOpt NVFP4/FP8 checkpoints are not yet fully "
                "supported for export. The weight-reconstruction core lives in "
                "mobius.integrations.modelopt (dequantize_nvfp4 / dequantize_fp8); "
                "remaining work is checkpoint weight-loading and native "
                "routed-expert NVFP4 QMoE emission (CUDA/Blackwell, "
                "onnxruntime_USE_FP4_QMOE=ON). Export the unquantized (bf16) "
                "checkpoint instead, or quantize the bf16 export via Olive."
            )
        # compressed-tensors is a container format, not an INT4 method. Its
        # ``bits``/``group_size`` values live inside ordered config groups and
        # may describe FP8 and NVFP4 simultaneously. Parsing it as this coarse
        # config would emit integer MatMulNBits nodes for FP4 E2M1 bytes.
        from mobius.integrations.compressed_tensors import (
            CompressedTensorsConfig,
            is_compressed_tensors_config,
        )

        if is_compressed_tensors_config(qc):
            CompressedTensorsConfig.parse(qc)
            return None
        # Block-scaled fp8 (E4M3 weight + 2D UE8M0 block scale) and packed-fp4
        # routed experts (I8-packed E2M1 nibbles + UE8M0 micro-scale) are a
        # mixed-precision layout this INT4/per-tensor path cannot load — the
        # packed [out, in/2] fp4 expert vs its logical [out, in] initializer
        # produces a confusing "Weight shape mismatch". Detect it by property
        # (not model name, not the ``quant_method`` string) and fail closed with
        # a typed, actionable blocker naming the real layout + the runtime ABI
        # gap. Checked before the ``quant_method == "none"`` early-return because
        # a checkpoint can advertise fp4 experts via top-level ``expert_dtype``
        # while leaving ``quant_method`` unset.
        from mobius.integrations._block_quant import BlockQuantScheme

        scheme = BlockQuantScheme.from_quantization_config(
            qc, expert_dtype=getattr(hf_config, "expert_dtype", None)
        )
        if scheme is not None:
            from mobius.integrations._block_quant import BlockQuantExportError

            raise BlockQuantExportError(
                "Block-scaled FP8 / packed-FP4 checkpoint is not loadable by the "
                "INT4/per-tensor quantization path. Detected "
                f"quant_method={scheme.quant_method!r}, "
                f"weight_block_size={list(scheme.weight_block_size) or None}, "
                f"expert_dtype={scheme.expert_dtype!r}: block-FP8 projections "
                "(E4M3 weight + 2D UE8M0 block scale) and/or FP4-packed routed "
                "experts (I8-packed E2M1 nibbles + UE8M0 micro-scale). Parse and "
                "validate these by property with mobius.integrations._block_quant "
                "(BlockQuantScheme / classify_tensor / QuantizedTensorDescriptor); "
                "the routed-expert emission gate (plan_routed_expert_bank) reports "
                "the exact onnx-genai nxrt ABI gap. Native export is blocked until "
                "the runtime gains a block-FP8 / planar-FP4 BlockFormat."
            )
        if method == "none":
            return None
        # Per-tensor fp8 (float8_e4m3fn + a scalar scale) is handled by dtype
        # casting in _assign_weight(), so it returns None here. (Block-scaled
        # fp8 was already routed to the typed blocker above.)
        if method == "fp8":
            return None
        return cls(
            bits=qc.get("bits", 4),
            group_size=qc.get("group_size", 128),
            quant_method=method,
            sym=qc.get("sym", qc.get("symmetric", True)),
            quantize_embeddings=bool(qc.get("embeds", False)),
            quantize_lm_head=bool(qc.get("lm_head", False)),
            quantize_vision=bool(qc.get("quantize_vision", False)),
            tie_word_embeddings=bool(qc.get("tie_word_embeddings", False)),
        )
