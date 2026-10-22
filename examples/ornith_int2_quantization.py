# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Experimental activation-diagonal scale fitting for Ornith's INT2 ABI."""

import torch


@torch.no_grad()
def activation_mse_int2(
    weight: torch.Tensor, diagonal: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unsigned codes and FP16 scales, retaining zero point 2/group 128.

    ``diagonal`` is the mean squared input activation, not a full Hessian.
    The RTN candidate is retained so the measured diagonal reconstruction loss
    cannot increase. This does not guarantee held-out or end-to-end improvement.
    """
    from olive.common.quant.utils import WeightQuantizer

    if weight.ndim != 2 or min(weight.shape) == 0 or weight.shape[1] % 128:
        raise ValueError("Expected a nonempty matrix with input width divisible by 128")
    if diagonal.shape != (weight.shape[1],):
        raise ValueError("Activation diagonal must match the projection input width")
    if not weight.is_floating_point() or not diagonal.is_floating_point():
        raise ValueError("Weights and activation diagonal must be floating point")
    if weight.device != diagonal.device:
        raise ValueError("Weights and activation diagonal must use the same device")
    if not torch.isfinite(weight).all() or not torch.isfinite(diagonal).all():
        raise ValueError("Weights and activation diagonal must be finite")
    if torch.any(diagonal < 0) or not torch.any(diagonal > 0):
        raise ValueError("Activation diagonal must be nonnegative with observed energy")
    quantizer = WeightQuantizer(bits=2, symmetric=True, group_size=128)
    minimum = torch.nextafter(
        torch.zeros((), dtype=torch.float16, device=weight.device),
        torch.ones((), dtype=torch.float16, device=weight.device),
    ).float()

    def rounded_scale(scale):
        rounded = scale.clamp_min(minimum).half().float()
        if not torch.isfinite(rounded).all():
            raise ValueError("INT2 scales overflow FP16")
        return rounded

    importance = (diagonal / diagonal.max()).float().reshape(1, -1, 128)
    all_codes, all_scales = [], []

    def loss(grouped, scale):
        q = (grouped / scale).round().clamp(-2, 1)
        return ((grouped - q * scale).square() * importance).sum(-1, keepdim=True)

    # Bound the search workspace independently of the projection output width.
    for chunk in weight.split(1024):
        w = chunk.float()
        grouped = w.reshape(w.shape[0], -1, 128)
        original_scale, _ = quantizer.find_qparams(w)
        original_scale = rounded_scale(original_scale).unsqueeze(-1)
        best_scale = original_scale.clone()

        best_loss = loss(grouped, best_scale)
        for ratio in torch.linspace(0.25, 1.25, 41, device=weight.device):
            candidate = rounded_scale(original_scale * ratio)
            candidate_loss = loss(grouped, candidate)
            better = candidate_loss < best_loss
            best_loss = torch.where(better, candidate_loss, best_loss)
            best_scale = torch.where(better, candidate, best_scale)
        for _ in range(4):
            q = (grouped / best_scale).round().clamp(-2, 1)
            denominator = (importance * q.square()).sum(-1, keepdim=True)
            fitted = (importance * grouped * q).sum(-1, keepdim=True)
            candidate = rounded_scale(
                torch.where(
                    denominator > 0,
                    fitted / denominator.clamp_min(torch.finfo(torch.float32).tiny),
                    best_scale,
                )
            )
            candidate_loss = loss(grouped, candidate)
            better = candidate_loss < best_loss
            best_loss = torch.where(better, candidate_loss, best_loss)
            best_scale = torch.where(better, candidate, best_scale)
        all_codes.append(
            ((grouped / best_scale).round().clamp(-2, 1) + 2).to(torch.int32).reshape(w.shape)
        )
        all_scales.append(best_scale.squeeze(-1).half())
    return torch.cat(all_codes), torch.cat(all_scales)
