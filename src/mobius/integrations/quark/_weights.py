# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Weight unpacking for AMD Quark INT4 checkpoints.

Quark stores each quantized projection as a packed payload on the canonical
``<proj>.weight`` key plus ``.weight_scale`` / ``.weight_zero_point`` sidecars,
with 8 ``uint4`` codes per int32 packed along the *output* axis in AWQ-GEMM
interleaved order. ``MatMulNBits`` wants two codes per byte along the *input*
axis in logical order, so the codes are fully unpacked and repacked rather than
reinterpreted in place.
"""

from __future__ import annotations

import math
from collections.abc import Collection

import torch

# AMD Quark packs 8 uint4 codes into one int32 in AWQ-GEMM interleaved order,
# so slot i of a packed word holds logical index ``_QUARK_NIBBLE_ORDER[i]``.
# Indexing a contiguous arange with this list undoes the interleave.
_QUARK_NIBBLE_ORDER: tuple[int, ...] = (0, 4, 1, 5, 2, 6, 3, 7)


def _unpack_quark_int32(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Expand ``[R, C // (32 // bits)]`` int32 into ``[R, C]`` unsigned codes.

    Mirrors ``quark.torch.utils.pack.Pack_4_bits.unpack(reorder=True)`` and the
    equivalent de-interleave in onnxruntime-genai's ``QuarkModel``.
    """
    if bits != 4:
        raise NotImplementedError(
            f"Quark unpacking is implemented for 4-bit weights only, got {bits}."
        )
    per_word = 32 // bits
    shifts = torch.arange(0, 32, bits, device=packed.device, dtype=torch.int32)
    # int32 -> one column per nibble slot, still in packed (interleaved) order.
    unpacked = (packed.unsqueeze(-1) >> shifts).reshape(packed.shape[0], -1)
    order = torch.arange(unpacked.shape[-1], device=packed.device)
    order = order.reshape(-1, per_word)[:, _QUARK_NIBBLE_ORDER].reshape(-1)
    return (unpacked[:, order] & ((1 << bits) - 1)).to(torch.uint8)


def _pack_nibbles_ort(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack ``[N, C]`` unsigned codes into ``[N, ceil(C*bits/8)]`` uint8.

    Two 4-bit codes per byte, low nibble first, matching what ``MatMulNBits``
    expects of its ``B`` and ``zero_points`` inputs. An odd trailing code is
    zero-padded into the high nibble of the final byte.
    """
    if bits != 4:
        raise NotImplementedError(
            f"Quark packing is implemented for 4-bit weights only, got {bits}."
        )
    rows, cols = codes.shape
    if cols % 2:
        codes = torch.cat(
            [codes, torch.zeros(rows, 1, dtype=codes.dtype, device=codes.device)], dim=1
        )
    pairs = codes.to(torch.uint8).reshape(rows, -1, 2)
    return (pairs[:, :, 0] & 0x0F) | ((pairs[:, :, 1] & 0x0F) << 4)


def preprocess_quark_weights(
    state_dict: dict[str, torch.Tensor],
    bits: int = 4,
    group_size: int = 32,
) -> dict[str, torch.Tensor]:
    """Convert AMD Quark packed weights into the ``MatMulNBits`` layout.

    Quark stores each quantized projection as three tensors::

        <proj>.weight             int32  [K, N // 8]   packed along N, interleaved
        <proj>.weight_scale       float  [K // gs, N]
        <proj>.weight_zero_point  int32  [K // gs, N // 8]

    and dequantizes as ``(code - zero_point) * scale``. Note the packing runs
    along the *output* axis, the transpose of the GPTQ/AWQ convention, so the
    codes are fully unpacked and repacked rather than reinterpreted in place.

    Unlike :func:`preprocess_awq_weights` no ``+1`` zero-point offset is applied:
    Quark stores true unbiased zero points, so they survive exactly (an AWQ
    round-trip cannot represent a zero point of 15).

    Args:
        state_dict: Weight dictionary containing Quark keys.
        bits: Quantization bit-width. Only 4 is supported.
        group_size: Number of input elements sharing one scale/zero point.

    Returns:
        State dict with ``<proj>.weight`` (uint8 ``[N, n_blocks, blob_size]``),
        ``<proj>.scales`` (``[N, n_blocks]``) and ``<proj>.zero_points``
        (uint8 ``[N, ceil(n_blocks*bits/8)]``) replacing the Quark keys.
    """
    blob_size = group_size * bits // 8
    # A Quark projection is identified by its scale sidecar: the packed payload
    # sits on the plain ``.weight`` key, which a float norm also uses.
    stems = {
        key[: -len(".weight_scale")] for key in state_dict if key.endswith(".weight_scale")
    }

    result: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        stem = key.rsplit(".", 1)[0]
        if stem in stems and key.endswith((".weight", ".weight_scale", ".weight_zero_point")):
            continue  # rebuilt below
        result[key] = value

    for stem in sorted(stems):
        packed = state_dict.get(f"{stem}.weight")
        scale = state_dict[f"{stem}.weight_scale"]
        if packed is None:
            raise ValueError(
                f"Missing {stem}.weight — the packed payload must accompany "
                f"{stem}.weight_scale."
            )

        n_groups, out_features = scale.shape
        in_features = packed.shape[0]
        if packed.shape[-1] * (32 // bits) != out_features:
            raise ValueError(
                f"{stem}.weight has shape {tuple(packed.shape)}, which does not pack "
                f"{out_features} output features at {bits} bits."
            )
        if math.ceil(in_features / group_size) != n_groups:
            raise ValueError(
                f"{stem}: {in_features} input features at group_size={group_size} "
                f"implies {math.ceil(in_features / group_size)} groups, but "
                f"{stem}.weight_scale has {n_groups}."
            )

        # [K, N] -> [N, K]; packing runs along the output axis in Quark.
        codes = _unpack_quark_int32(packed, bits).t().contiguous()
        result[f"{stem}.weight"] = _pack_nibbles_ort(codes, bits).reshape(
            out_features, n_groups, blob_size
        )
        result[f"{stem}.scales"] = scale.t().contiguous()

        packed_zeros = state_dict.get(f"{stem}.weight_zero_point")
        if packed_zeros is not None:
            zeros = _unpack_quark_int32(packed_zeros, bits).t().contiguous()
            result[f"{stem}.zero_points"] = _pack_nibbles_ort(zeros, bits)

    return result


def dequantize_quark_weight(
    packed: torch.Tensor,
    scale: torch.Tensor,
    packed_zeros: torch.Tensor | None,
    *,
    bits: int,
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Reconstruct a dense ``[out_features, in_features]`` float weight.

    Applies Quark's ``(code - zero_point) * scale`` per group. Needed where a
    model's ONNX graph uses a plain float ``Linear`` for a projection the
    checkpoint happened to quantize, so the packed sidecars can never bind to a
    ``MatMulNBits`` initializer (Gemma4's per-layer-embedding projections are
    the motivating case).
    """
    codes = _unpack_quark_int32(packed, bits).t()  # [N, K]
    out_features, in_features = codes.shape
    n_groups = scale.shape[0]
    if in_features % n_groups:
        raise ValueError(f"{in_features} input features do not divide into {n_groups} groups.")
    grouped = codes.reshape(out_features, n_groups, in_features // n_groups).to(torch.float32)
    if packed_zeros is not None:
        zeros = _unpack_quark_int32(packed_zeros, bits).t().to(torch.float32)
        grouped = grouped - zeros.unsqueeze(-1)
    scaled = grouped * scale.to(torch.float32).t().unsqueeze(-1)
    return scaled.reshape(out_features, in_features).to(dtype)


def dequantize_quark_projections(
    state_dict: dict[str, torch.Tensor],
    stems: Collection[str],
    *,
    bits: int,
    group_size: int,
    dtype: torch.dtype,
) -> None:
    """Replace named Quark projections with dense float weights, in place.

    Silently skips stems that are absent or already float, so a caller can name
    every projection its graph keeps float without first inspecting the
    checkpoint.
    """
    for stem in stems:
        scale = state_dict.get(f"{stem}.weight_scale")
        packed = state_dict.get(f"{stem}.weight")
        if scale is None or packed is None:
            continue
        state_dict[f"{stem}.weight"] = dequantize_quark_weight(
            packed,
            scale,
            state_dict.get(f"{stem}.weight_zero_point"),
            bits=bits,
            group_size=group_size,
            dtype=dtype,
        )
        state_dict.pop(f"{stem}.weight_scale", None)
        state_dict.pop(f"{stem}.weight_zero_point", None)

