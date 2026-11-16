# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generate complete GNM meshes using the independently pinned upstream CUDA model.

Install the reference from google/gnm at GNM_REFERENCE_REVISION, then run
``python scripts/generate_gnm_golden.py --reference-root <checkout>``.
This script never invokes a Mobius model or graph to produce reference outputs.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from mobius._testing.gnm import (
    GNM_GOLDEN_DIR,
    GNM_REFERENCE_REVISION,
    gnm_inputs,
    gnm_semantic_inputs,
)
from mobius.models.gnm import GNM_MODEL_ID, GNM_REVISION, GNM_SHA256


def generate(
    reference_root: Path,
    levels: list[str] | None = None,
    output_dir: Path = GNM_GOLDEN_DIR,
    case_id: str = "gnm-v3",
    device: str = "cuda",
) -> None:
    """Generate full reference arrays without invoking a Mobius model implementation."""
    root = reference_root.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != GNM_REFERENCE_REVISION:
        raise RuntimeError(
            f"GNM reference is at {revision}; check out {GNM_REFERENCE_REVISION}."
        )
    if subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ):
        raise RuntimeError("GNM reference checkout is modified; use a clean pinned checkout.")
    sys.path.insert(0, str(root))
    from gnm.shape import gnm_pytorch

    if not Path(gnm_pytorch.__file__).resolve().is_relative_to(root):
        raise RuntimeError(
            "Imported GNM is not from --reference-root; remove conflicting imports."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("GNM reference goldens require CUDA; select an idle GPU.")
    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise ValueError("GNM reference goldens require a CUDA device, e.g. cuda:0.")
    if target_device.index is None:
        target_device = torch.device("cuda", torch.cuda.current_device())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = Path(
        hf_hub_download(GNM_MODEL_ID, "v3_0/gnm_head.npz", revision=GNM_REVISION)
    )
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != GNM_SHA256:
        raise RuntimeError("GNM archive checksum differs; delete the cached file and retry.")
    reference = gnm_pytorch.GNM.from_custom_file(checkpoint).to(target_device)
    output_dir.mkdir(parents=True, exist_ok=True)
    for level in levels or ["L4", "L5"]:
        feeds = gnm_inputs(level)
        with torch.no_grad(), torch.device(target_device):
            vertices = (
                reference(
                    **{
                        name: torch.from_numpy(value).to(target_device)
                        for name, value in feeds.items()
                    }
                )
                .cpu()
                .numpy()
            )
        assert vertices.shape == (len(feeds["identity"]), 17821, 3)
        assert np.isfinite(vertices).all()
        suffix = "" if level == "L4" else "_generation"
        target = output_dir / f"{case_id}{suffix}"
        additional = {}
        if level == "L5":
            with torch.no_grad(), torch.device(target_device):
                additional["semantic_vertices"] = (
                    reference(
                        **{
                            name: torch.from_numpy(value).to(target_device)
                            for name, value in gnm_semantic_inputs().items()
                        }
                    )
                    .cpu()
                    .numpy()
                )
            with np.load(checkpoint, allow_pickle=False) as archive:
                names = archive["vertex_group_names"].tolist()
                for group in ("left_eye", "right_eye"):
                    additional[group] = archive["vertex_groups"][names.index(group)] > 0.5
        np.savez_compressed(
            target.with_suffix(".npz"), vertices=vertices, **feeds, **additional
        )
        metadata = {
            "model_id": GNM_MODEL_ID,
            "revision": GNM_REVISION,
            "checkpoint_sha256": GNM_SHA256,
            "reference_revision": revision,
            "reference_device": str(target_device),
            "device_name": torch.cuda.get_device_name(target_device),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "fixture_sha256": hashlib.sha256(
                target.with_suffix(".npz").read_bytes()
            ).hexdigest(),
            "level": level,
            "shape": list(vertices.shape),
            "contract": "full articulated mesh; no vertices, basis rows or joints pruned",
        }
        target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(
            f"{level}: {vertices.shape}, reference={revision}, "
            f"CUDA={torch.cuda.get_device_name(target_device)}"
        )
