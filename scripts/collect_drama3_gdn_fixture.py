# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Collect private reduced trained-weight GDN fixtures from pinned local assets.

No downloads, GPU execution, source edits, Mobius inference, or monkeypatches.
The checkpoint has its own license: generated weights belong outside the MIT
repository. Run with the pinned Torch 2.8 reference environment and preserve
the resulting manifest with numerical reports. This is not a full checkpoint
or a full-model oracle.
"""

from __future__ import annotations
import __future__

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as functional
from safetensors import safe_open

from mobius._configs.fish_omni import DRAMA3_REVISION, DRAMA3_SOURCE_REVISION

LAYERS_SHA256 = "e4704b4fde7906bf021d6696c4132458b4bc52c836d22da702275252683540bc"
LENGTHS = (1, 2, 30, 64, 65, 128, 129, 257)


def source_namespace(path):
    """Compile only unmodified, dependency-closed source definitions."""
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != LAYERS_SHA256:
        raise ValueError("Pinned Fish layers source hash mismatch")
    names = {
        "GatedRMSNorm",
        "l2norm",
        "chunk_gated_delta_rule",
        "recurrent_gated_delta_rule",
        "GatedDeltaNet",
    }
    definitions = [
        node
        for node in ast.parse(data).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    if {node.name for node in definitions} != names:
        raise ValueError("Pinned Fish GDN definitions missing")
    namespace = {"torch": torch, "nn": torch.nn, "F": functional, "GDN_CHUNK_SIZE": 64}
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace


def collect(checkpoint, source_dir, output):
    if not torch.__version__.startswith("2.8."):
        raise ValueError("Fixture reference requires pinned Torch 2.8")
    if checkpoint.name != DRAMA3_REVISION or source_dir.name != DRAMA3_SOURCE_REVISION:
        raise ValueError("Expected immutable pinned checkpoint and source directories")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    namespace = source_namespace(source_dir / "fish_omni.layers.py")
    index_path = checkpoint / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    config = json.loads((checkpoint / "config.json").read_text())["text_config"]
    key_dim = config["linear_num_key_heads"] * config["linear_key_head_dim"]
    assert config["linear_key_head_dim"] == config["linear_value_head_dim"] == 128
    prefix = "model.language_model.layers.0.linear_attn."
    manifest = {
        "scope": "Reduced trained-weight layer0 GDN; hidden32, key head0, value heads0:2, head widths128. Not full checkpoint/model or CUDA bit parity.",
        "checkpoint": DRAMA3_REVISION,
        "source": DRAMA3_SOURCE_REVISION,
        "layers_sha256": LAYERS_SHA256,
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest(),
        "license_sha256": hashlib.sha256((checkpoint / "LICENSE").read_bytes()).hexdigest(),
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "license": "Fish Audio Community License; separate from Mobius MIT. This collector does not accept terms on the user's behalf.",
        "torch": torch.__version__,
        "device": "cpu",
        "weights": {},
        "cases": [],
    }
    weights = {}

    def read(name, selections):
        source_name = prefix + name
        with safe_open(
            checkpoint / weight_map[source_name], framework="pt", device="cpu"
        ) as f:
            tensor = f.get_slice(source_name)
            pieces = [tensor[selection] for selection in selections]
            array = torch.cat(pieces, dim=0).float().numpy()
            manifest["weights"][name] = {
                "source_name": source_name,
                "source_shape": tensor.get_shape(),
                "slices": [repr(selection) for selection in selections],
                "float32_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        weights[name] = array

    qkv_rows = (
        slice(0, 128),
        slice(key_dim, key_dim + 128),
        slice(2 * key_dim, 2 * key_dim + 256),
    )
    read("in_proj_qkv.weight", [(rows, slice(0, 32)) for rows in qkv_rows])
    read("conv1d.weight", [(rows, slice(None), slice(None)) for rows in qkv_rows])
    for name in ("in_proj_z.weight",):
        read(name, [(slice(0, 256), slice(0, 32))])
    for name in ("in_proj_a.weight", "in_proj_b.weight"):
        read(name, [(slice(0, 2), slice(0, 32))])
    for name in ("A_log", "dt_bias"):
        read(name, [slice(0, 2)])
    read("norm.weight", [slice(None)])
    read("out_proj.weight", [(slice(0, 32), slice(0, 256))])
    embedding_name = "model.language_model.embed_tokens.weight"
    with safe_open(checkpoint / weight_map[embedding_name], framework="pt", device="cpu") as f:
        tensor = f.get_slice(embedding_name)
        embeddings = (
            torch.stack([tensor[start : start + 287, :32] for start in (0, 512)])
            .float()
            .numpy()
        )
    manifest["inputs"] = {
        "source_name": embedding_name,
        "rows": "[0:287] and [512:799]; columns [0:32]",
        "float32_sha256": hashlib.sha256(embeddings.tobytes()).hexdigest(),
    }
    np.savez_compressed(output / "weights.npz", **weights, embeddings=embeddings)
    reduced = SimpleNamespace(
        hidden_size=32,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        rms_norm_eps=config["rms_norm_eps"],
    )
    for dtype_name, dtype in (
        ("f32", torch.float32),
        ("f16", torch.float16),
        ("bf16", torch.bfloat16),
    ):
        module = namespace["GatedDeltaNet"](reduced, dtype=dtype, device="cpu")
        module.load_state_dict(
            {name: torch.from_numpy(value).to(dtype) for name, value in weights.items()}
        )
        for batch in (1, 2):
            source_inputs = torch.from_numpy(embeddings[:batch]).to(dtype)
            for length in LENGTHS:
                for continuing in (False, True):
                    module.allocate_cache(batch, dtype, "cpu")
                    with torch.inference_mode():
                        if continuing:
                            module(
                                source_inputs[:, :30],
                                SimpleNamespace(continue_state=False, use_cache=True),
                            )
                        else:
                            # Nonzero stale buffers must be ignored, not accidentally pass as zeros.
                            module.conv_state.fill_(0.375)
                            module.recurrent_state.fill_(-0.125)
                        conv = module.conv_state.clone()
                        state = module.recurrent_state.clone()
                        start = 30 if continuing else 0
                        x = source_inputs[:, start : start + length]
                        result = module(
                            x, SimpleNamespace(continue_state=continuing, use_cache=True)
                        )
                    filename = f"{dtype_name}-b{batch}-t{length}-c{int(continuing)}.npz"
                    np.savez_compressed(
                        output / filename,
                        x=x.float().numpy(),
                        conv=conv.float().numpy(),
                        state=state.numpy(),
                        output=result.float().numpy(),
                        present_conv=module.conv_state.float().numpy(),
                        present_state=module.recurrent_state.numpy(),
                    )
                    manifest["cases"].append(
                        {
                            "file": filename,
                            "dtype": dtype_name,
                            "batch": batch,
                            "length": length,
                            "continuing": continuing,
                            "rule": "recurrent" if continuing and length == 1 else "chunk",
                            "sha256": hashlib.sha256(
                                (output / filename).read_bytes()
                            ).hexdigest(),
                        }
                    )
    manifest["weights_file_sha256"] = hashlib.sha256(
        (output / "weights.npz").read_bytes()
    ).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "fixture": str(output),
                "cases": len(manifest["cases"]),
                "scope": manifest["scope"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.checkpoint, args.source_dir, args.output)
