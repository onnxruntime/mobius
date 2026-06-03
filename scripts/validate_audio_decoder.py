"""Per-codebook numerical parity test: mobius audio_decoder.onnx vs HF depthformer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort
from safetensors.torch import load_file

# HF model classes
from liquid_audio.model.lfm2_audio import (
    LFM2AudioConfig,
    LFM2AudioModel,
    DepthformerConfig,
)
from liquid_audio.model.conformer.encoder import ConformerEncoderConfig
from liquid_audio.processor import PreprocessorConfig
from transformers import Lfm2Config


CKPT = Path(
    "/home/justinchu/.cache/huggingface/hub/models--LiquidAI--LFM2-Audio-1.5B/snapshots/c798aad30dc3cd72e72970beab51326b8443bd94"
)
ONNX_PATH = "/tmp/lfm2-out/audio_decoder.onnx"


def build_hf_model() -> LFM2AudioModel:
    cfg = json.loads((CKPT / "config.json").read_text())
    conf = LFM2AudioConfig(
        lfm=Lfm2Config(**cfg.pop("lfm")),
        encoder=ConformerEncoderConfig(**cfg.pop("encoder")),
        depthformer=DepthformerConfig(**cfg.pop("depthformer")),
        preprocessor=PreprocessorConfig(**cfg.pop("preprocessor")),
        **cfg,
    )
    # Build on CPU, fp32, no flash-attn.
    model = LFM2AudioModel(conf)
    model.lfm.set_attn_implementation("sdpa")
    sd = load_file(str(CKPT / "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"HF load: missing={len(missing)} unexpected={len(unexpected)}")
    # Only freqs_cis-like buffers should be missing; print first few of each.
    if missing:
        print("  missing examples:", missing[:5])
    if unexpected:
        print("  unexpected examples:", unexpected[:5])
    model.eval().float()
    return model


def hf_single_step(
    model: LFM2AudioModel,
    backbone_hidden: torch.Tensor,  # (1, 1, 2048)
    prev_embedding: torch.Tensor,   # (1, 1, 1024) — added as depthformer_token
    codebook_idx: int,
):
    """Replicate one step of the HF generation loop for the given codebook.

    Mirrors lfm2_audio.py L508–L532. backbone_hidden plays the role of `embedding`,
    prev_embedding plays the role of `depthformer_token`. Returns (logits, cache).
    """
    # Squeeze to match HF's `embedding` shape (hidden,)
    emb = backbone_hidden.reshape(-1)  # (2048,)
    depthformer_in = model.depth_linear(emb)  # (C * D,)
    depthformer_in = depthformer_in.view(model.codebooks, model.depthformer_dim)
    depthformer_token = prev_embedding.reshape(model.depthformer_dim)  # (D,)
    cur_in = depthformer_in[codebook_idx] + depthformer_token  # (D,)
    cur_in = cur_in[None, None, :]  # (1, 1, D)
    out, cache = model.depthformer.forward_cached(cur_in, None)
    logits = model.depth_embeddings[codebook_idx].get_logits(out.squeeze(0).squeeze(0))
    # logits shape: (vocab,)
    return logits, cache


def main():
    print("Loading HF model …")
    hf = build_hf_model()
    print("Loading ONNX session …")
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]
    n_layers = sum(1 for n in in_names if n.startswith("past_key_values.") and n.endswith(".key"))
    print(f"ONNX layers: {n_layers}")

    torch.manual_seed(0)
    backbone_hidden = torch.randn(1, 1, 2048)
    torch.manual_seed(1)
    prev_embedding = torch.randn(1, 1, 1024)

    empty_kv = np.zeros((1, 8, 0, 32), dtype=np.float32)

    results = {}
    for ci in range(8):
        # HF reference
        with torch.no_grad():
            hf_logits, hf_cache = hf_single_step(hf, backbone_hidden, prev_embedding, ci)
        hf_logits_np = hf_logits.cpu().numpy()

        # ONNX
        feeds = {
            "backbone_hidden": backbone_hidden.cpu().numpy(),
            "prev_embedding": prev_embedding.cpu().numpy(),
            "codebook_idx": np.array(ci, dtype=np.int64),
        }
        for li in range(n_layers):
            feeds[f"past_key_values.{li}.key"] = empty_kv
            feeds[f"past_key_values.{li}.value"] = empty_kv
        outs = sess.run(out_names, feeds)
        out_map = dict(zip(out_names, outs))
        onnx_logits = out_map["codebook_logits"].reshape(-1)

        # Compare logits
        diff = np.abs(onnx_logits - hf_logits_np)
        max_d = float(diff.max())
        mean_d = float(diff.mean())
        med_d = float(np.median(diff))
        passed = np.allclose(onnx_logits, hf_logits_np, atol=1e-3, rtol=1e-3)
        results[ci] = (max_d, mean_d, med_d, passed)
        print(
            f"cb={ci}  shape_hf={hf_logits_np.shape} shape_onnx={onnx_logits.shape} "
            f"max={max_d:.3e} mean={mean_d:.3e} median={med_d:.3e}  pass={passed}"
        )

        # KV cache compare for layer 0 (just sanity-check shapes + a few values)
        if ci == 0:
            for li in range(n_layers):
                hf_k, hf_v = hf_cache[li]
                # HF cache shape: (B, T, gqa_dim, head_dim) per LayerKVCache.update
                # Transpose to (B, gqa_dim, T, head_dim) to match ONNX layout.
                hf_k_t = hf_k.transpose(1, 2).cpu().numpy()
                hf_v_t = hf_v.transpose(1, 2).cpu().numpy()
                onnx_k = out_map[f"present.{li}.key"]
                onnx_v = out_map[f"present.{li}.value"]
                k_max = float(np.abs(onnx_k - hf_k_t).max())
                v_max = float(np.abs(onnx_v - hf_v_t).max())
                print(
                    f"  layer{li}: K shape hf={hf_k_t.shape} onnx={onnx_k.shape} max_d_k={k_max:.3e}  "
                    f"V max_d_v={v_max:.3e}"
                )

    print("\n=== SUMMARY ===")
    all_pass = all(r[3] for r in results.values())
    for ci, (mx, mn, md, p) in results.items():
        print(f"  codebook {ci}: max={mx:.3e}  mean={mn:.3e}  median={md:.3e}  {'PASS' if p else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'} (atol=rtol=1e-3)")


if __name__ == "__main__":
    main()
