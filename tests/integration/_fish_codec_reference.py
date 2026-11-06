# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test-only, immutable upstream codec reference with no optional audio imports.

Only the unused AudioSignal/BaseModel/CodecMixin imports are stubbed. All
decoder, RVQ, weight-normalization, Snake, transformer and ConvNeXt math runs
unmodified upstream source. No source or checkpoint is redistributed.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import urllib.request
from pathlib import Path
from unittest.mock import patch

import torch

from mobius._configs.fish_codec import FISH_CODEC_SOURCE_REVISION, FishCodecConfig

_DAC_REVISION = "c7cfc5d2647e26471dc394f95846a0830e7bec34"


def load_reference_sources(directory: Path):
    sources = {
        "dac.nn.layers": (
            "descriptinc/descript-audio-codec",
            _DAC_REVISION,
            "dac/nn/layers.py",
        ),
        "dac.nn.quantize": (
            "descriptinc/descript-audio-codec",
            _DAC_REVISION,
            "dac/nn/quantize.py",
        ),
        "_mobius_fish_codec_dac": (
            "fishaudio/fish-speech",
            FISH_CODEC_SOURCE_REVISION,
            "fish_speech/models/dac/modded_dac.py",
        ),
        "_mobius_fish_codec_rvq": (
            "fishaudio/fish-speech",
            FISH_CODEC_SOURCE_REVISION,
            "fish_speech/models/dac/rvq.py",
        ),
    }
    paths = {}
    for name, (repo, revision, path) in sources.items():
        target = directory / (name.replace(".", "_") + ".py")
        urllib.request.urlretrieve(
            f"https://raw.githubusercontent.com/{repo}/{revision}/{path}", target
        )
        paths[name] = target

    stubs = {
        name: types.ModuleType(name)
        for name in ("audiotools", "audiotools.ml", "dac.model.base")
    }
    stubs["audiotools"].AudioSignal = object
    stubs["audiotools.ml"].BaseModel = torch.nn.Module
    stubs["dac.model.base"].CodecMixin = type("UnusedCodecMixin", (), {})
    modules = {}
    with patch.dict(sys.modules, stubs):
        for name, path in paths.items():
            spec = importlib.util.spec_from_file_location(name, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            # dataclasses and TorchScript need the real source-backed module.
            with patch.dict(sys.modules, {**modules, name: module}):
                spec.loader.exec_module(module)
            modules[name] = module
    return modules["_mobius_fish_codec_dac"], modules["_mobius_fish_codec_rvq"]


def make_reference(config: FishCodecConfig, sources):
    dac, rvq = sources

    class Reference(torch.nn.Module):
        def __init__(self):
            super().__init__()
            args = dac.ModelArgs(
                dim=config.latent_dim,
                n_head=config.transformer_heads,
                n_local_heads=config.transformer_heads,
                head_dim=config.head_dim,
                n_layer=config.transformer_layers,
                intermediate_size=config.transformer_intermediate_size,
                norm_eps=config.norm_eps,
                rope_base=config.rope_base,
            )
            post = dac.WindowLimitedTransformer(
                args,
                input_dim=config.latent_dim,
                window_size=config.transformer_window,
                causal=True,
            )
            self.quantizer = rvq.DownsampleResidualVectorQuantize(
                input_dim=config.latent_dim,
                n_codebooks=config.residual_codebooks,
                codebook_size=config.residual_codebook_size,
                semantic_codebook_size=config.semantic_codebook_size,
                codebook_dim=config.codebook_dim,
                downsample_factor=config.quantizer_rates,
                post_module=post,
            )
            self.decoder = dac.Decoder(
                config.latent_dim,
                config.decoder_dim,
                config.decoder_rates,
                causal=True,
                n_transformer_layers=[0] * len(config.decoder_rates),
            )

        def forward(self, codes):
            # Upstream mutates/clamps indices in place. Do not mutate test feeds.
            return dac.DAC.from_indices(self, codes.clone())

    # Avoid allocating the upstream 32768-square unused causal_mask. The
    # WindowLimitedTransformer.forward builds its own actual attention mask.
    with torch.device("meta"):
        model = Reference()
    model.quantizer.post_module.causal_mask = torch.empty(0, device="meta")
    model.quantizer.post_module.freqs_cis = torch.empty(0, device="meta")
    model.to_empty(device="cpu")
    model.quantizer.post_module.freqs_cis = dac.precompute_freqs_cis(
        config.max_frames, config.head_dim, config.rope_base
    )
    return model.eval()


def initialize_reference(model):
    """Deterministic, well-conditioned weights for production-topology tiny tests."""
    generator = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(".alpha"):
                param.uniform_(0.5, 1.5, generator=generator)
            elif name.endswith("norm.weight"):
                param.fill_(1)
            elif name.endswith((".weight_g", ".original0")):
                param.fill_(0.2)
            else:
                param.normal_(0, 0.05, generator=generator)
    return model
