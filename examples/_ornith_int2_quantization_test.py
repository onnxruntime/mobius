# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import torch
from ornith_int2_quantization import activation_mse_int2


def test_activation_mse_preserves_layout_and_does_not_increase_diagonal_loss():
    from olive.common.quant.utils import WeightQuantizer

    generator = torch.Generator().manual_seed(42)
    weight = torch.randn(131, 256, generator=generator) * 0.02
    diagonal = torch.rand(256, generator=generator).square()
    quantizer = WeightQuantizer(bits=2, symmetric=True, group_size=128)
    scale, zero = quantizer.find_qparams(weight)
    scale = scale.half().float()
    rtn = quantizer.dequantize(quantizer.quantize(weight, scale, zero), scale, zero)
    codes, scales = activation_mse_int2(weight, diagonal)
    assert codes.shape == weight.shape and codes.dtype == torch.int32
    assert scales.shape == (131, 2) and scales.dtype == torch.float16
    assert codes.min() >= 0 and codes.max() <= 3
    actual = (codes.float() - 2) * scales.float().repeat_interleave(128, -1)
    loss = ((actual - weight).square() * diagonal).reshape(131, 2, 128).sum(-1)
    baseline = ((rtn - weight).square() * diagonal).reshape(131, 2, 128).sum(-1)
    assert torch.all(loss <= baseline + 1e-8)


def test_activation_mse_preserves_zero_and_subnormal_scales():
    weight = torch.stack([torch.zeros(128), torch.full((128,), 1e-22)])
    codes, scales = activation_mse_int2(weight, torch.ones(128))
    assert torch.all(codes == 2)
    assert torch.all(torch.isfinite(scales) & (scales > 0))


def test_activation_mse_preserves_chunk_boundary_and_diagonal_scale_invariance():
    generator = torch.Generator().manual_seed(21)
    weight = torch.randn(1025, 128, generator=generator) * 0.01
    diagonal = torch.rand(128, generator=generator)
    codes, scales = activation_mse_int2(weight, diagonal)
    tail_codes, tail_scales = activation_mse_int2(weight[-1:], diagonal)
    assert torch.equal(codes[-1:], tail_codes)
    assert torch.equal(scales[-1:], tail_scales)
    scaled_codes, scaled_scales = activation_mse_int2(weight[-1:], diagonal.double() * 1e100)
    assert torch.equal(tail_codes, scaled_codes)
    assert torch.equal(tail_scales, scaled_scales)


@pytest.mark.parametrize(
    "weight,diagonal",
    [
        (torch.ones(2, 127), torch.ones(127)),
        (torch.ones(0, 128), torch.ones(128)),
        (torch.ones(2, 0), torch.ones(0)),
        (torch.ones(2, 128), torch.ones(127)),
        (torch.ones(2, 128), torch.zeros(128)),
        (torch.ones(2, 128), -torch.ones(128)),
        (torch.full((2, 128), float("nan")), torch.ones(128)),
        (torch.ones(2, 128), torch.full((128,), float("inf"))),
        (torch.ones(2, 128, dtype=torch.int32), torch.ones(128)),
        (torch.full((2, 128), 1e10), torch.ones(128)),
    ],
)
def test_activation_mse_rejects_invalid_inputs(weight, diagonal):
    with pytest.raises(ValueError):
        activation_mse_int2(weight, diagonal)
