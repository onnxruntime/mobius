"""Bisect per-layer divergence between HF Conformer and the mobius math.

Hooks every HF ConformerLayer's output, runs a torch re-impl of mobius math
(loaded with the same weights), and reports first layer where they differ.

Also dumps sub-block (ff1/attn/conv/ff2/norm_out) deltas for the first bad
layer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from _mobius_torch_conformer import MobEncoder, load_from_hf  # noqa: E402

from liquid_audio import LFM2AudioModel  # noqa: E402
from liquid_audio.processor import LFM2AudioProcessor  # noqa: E402

REPO = "LiquidAI/LFM2-Audio-1.5B"
AUDIO_PATH = "testdata/652-129742-0006.flac"
TARGET_SECONDS = 1.0


def main() -> int:
    torch.manual_seed(0)
    print("Loading HF model...", flush=True)
    model = LFM2AudioModel.from_pretrained(REPO, dtype=torch.float32, device="cpu")
    model.eval()
    proc = LFM2AudioProcessor.from_pretrained(REPO, device="cpu")
    proc.audio_processor.to(dtype=torch.float32)

    audio, sr = sf.read(AUDIO_PATH, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n = int(sr * TARGET_SECONDS)
    audio = audio[:n]
    wav = torch.from_numpy(audio).unsqueeze(0).to(torch.float32)
    lengths = torch.tensor([wav.shape[1]], dtype=torch.long)
    with torch.no_grad():
        mel, mel_lens = proc.audio_processor(wav, lengths)
    print(f"Mel: {tuple(mel.shape)}  lens={mel_lens.tolist()}")

    # ------------------------------------------------------------------
    # 1. HF forward, hooking each layer input/output (in [B, T, D] space).
    # ------------------------------------------------------------------
    hf = model.conformer
    hf_layer_inputs: list[torch.Tensor] = []
    hf_layer_outputs: list[torch.Tensor] = []

    def make_in_hook(idx):
        def hook(_mod, args, kwargs):
            x = kwargs.get('x', args[0] if args else None)
            hf_layer_inputs.append(x.detach().clone())
        return hook

    def make_out_hook(idx):
        def hook(_mod, _args, output):
            out = output[0] if isinstance(output, tuple) else output
            hf_layer_outputs.append(out.detach().clone())
        return hook

    handles = []
    for i, layer in enumerate(hf.layers):
        handles.append(layer.register_forward_pre_hook(make_in_hook(i), with_kwargs=True))
        handles.append(layer.register_forward_hook(make_out_hook(i)))

    # Also hook pre_encode output, attn input/output, conv input/output etc. for first layer
    sub_dumps = {}

    layer0 = hf.layers[0]

    def hook_capture(name):
        def h(_m, _a, out):
            o = out[0] if isinstance(out, tuple) else out
            sub_dumps[name] = o.detach().clone()
        return h

    handles.append(layer0.norm_feed_forward1.register_forward_hook(hook_capture('norm_ff1')))
    handles.append(layer0.feed_forward1.register_forward_hook(hook_capture('ff1')))
    handles.append(layer0.norm_self_att.register_forward_hook(hook_capture('norm_sa')))
    handles.append(layer0.self_attn.register_forward_hook(hook_capture('attn')))
    handles.append(layer0.norm_conv.register_forward_hook(hook_capture('norm_conv')))
    handles.append(layer0.conv.register_forward_hook(hook_capture('conv')))
    handles.append(layer0.norm_feed_forward2.register_forward_hook(hook_capture('norm_ff2')))
    handles.append(layer0.feed_forward2.register_forward_hook(hook_capture('ff2')))
    handles.append(layer0.norm_out.register_forward_hook(hook_capture('norm_out')))

    with torch.no_grad():
        hf_out, hf_lens = hf(mel, mel_lens)
    for h in handles:
        h.remove()

    print(f"HF: {len(hf_layer_outputs)} layers captured")

    # ------------------------------------------------------------------
    # 2. Build mobius torch reimpl, load weights, run with sub-block dumps.
    # ------------------------------------------------------------------
    mob = MobEncoder(
        n_mels=128, d_model=512, num_heads=8, d_inner=2048,
        num_layers=17, k=9, c=256,
    )
    load_from_hf(mob, hf)
    mob.eval()

    layer_dumps: list[dict] = []
    with torch.no_grad():
        # mob expects [B, T, n_mels]; mel is [B, n_mels, T]
        mob_input = mel.transpose(1, 2)
        mob_out = mob(mob_input, layer_dumps=layer_dumps)

    # Compare pre_encode output to HF input to layer 0
    with torch.no_grad():
        pre = mob.pre_encode(mob_input)
    d = (pre - hf_layer_inputs[0]).abs()
    print(f"\npre_encode  max={d.max().item():.3e}  mean={d.mean().item():.3e}")

    # Per-layer comparison
    print("\n=== Per-layer max|diff| (HF vs mobius-torch) ===")
    first_bad = None
    for i, (hf_o, mob_d) in enumerate(zip(hf_layer_outputs, layer_dumps)):
        mob_o = mob_d['after_norm_out']
        d = (hf_o - mob_o).abs()
        flag = ""
        if d.max().item() > 1e-3 and first_bad is None:
            first_bad = i
            flag = "  <-- first >1e-3"
        print(f"layer {i:2d}: max={d.max().item():.3e}  mean={d.mean().item():.3e}{flag}")

    # Sub-block deltas for layer 0
    print("\n=== Sub-block deltas for layer 0 ===")
    d0 = layer_dumps[0]
    pairs = [
        ('ff1 (output of feed_forward1)', sub_dumps['ff1'], d0['after_ff1'] - d0['in']),  # mob: x + 0.5*ff1 - x = 0.5*ff1 ... no
    ]
    # Simpler: just compare each captured tensor name
    print("Note: HF 'after_X' = residual after each block; we'll reconstruct.")
    # Reconstruct HF sub-block outputs
    x = hf_layer_inputs[0]
    after_ff1 = x + 0.5 * sub_dumps['ff1']
    after_attn = after_ff1 + sub_dumps['attn']
    after_conv = after_attn + sub_dumps['conv']
    after_ff2 = after_conv + 0.5 * sub_dumps['ff2']

    def cmp(name, a, b):
        d = (a - b).abs()
        print(f"  {name:30s}  max={d.max().item():.3e}  mean={d.mean().item():.3e}")

    cmp('layer-in (pre_encode→layer0)', hf_layer_inputs[0], d0['in'])
    cmp('ff1 raw output', sub_dumps['ff1'], d0['after_ff1'] - d0['in'] * 1.0 - (d0['after_ff1'] - d0['in']) + 2*(d0['after_ff1'] - d0['in']))  # placeholder; recompute next
    # Actually just rerun mob layer with intermediate dumps of raw block outputs.
    print("\n  (rerunning mob layer 0 with raw sub-block outputs)")
    with torch.no_grad():
        l0 = mob.layers[0]
        xi = hf_layer_inputs[0]
        ff1_mob = l0.feed_forward1(l0.norm_feed_forward1(xi))
        x1 = xi + 0.5 * ff1_mob
        attn_mob = l0.self_attn(l0.norm_self_att(x1))
        x2 = x1 + attn_mob
        conv_mob = l0.conv(l0.norm_conv(x2))
        x3 = x2 + conv_mob
        ff2_mob = l0.feed_forward2(l0.norm_feed_forward2(x3))
        x4 = x3 + 0.5 * ff2_mob
        out_mob = l0.norm_out(x4)

        # Same for HF with shared input xi
        h_l0 = hf.layers[0]
        # Recompute HF block raw outputs by feeding xi
        h_ff1 = h_l0.feed_forward1(h_l0.norm_feed_forward1(xi))
        h_x1 = xi + 0.5 * h_ff1
        # HF attn needs pos_emb
        _, pos_emb = hf.pos_enc(x=xi, cache_len=0)
        h_attn = h_l0.self_attn(query=h_l0.norm_self_att(h_x1), key=h_l0.norm_self_att(h_x1), value=h_l0.norm_self_att(h_x1), mask=None, pos_emb=pos_emb, cache=None)
        h_x2 = h_x1 + h_attn
        h_conv = h_l0.conv(h_l0.norm_conv(h_x2), pad_mask=None, cache=None)
        h_x3 = h_x2 + h_conv
        h_ff2 = h_l0.feed_forward2(h_l0.norm_feed_forward2(h_x3))
        h_x4 = h_x3 + 0.5 * h_ff2
        h_out = h_l0.norm_out(h_x4)

    cmp('ff1 raw',  ff1_mob,  h_ff1)
    cmp('after ff1', x1, h_x1)
    cmp('attn raw', attn_mob, h_attn)
    cmp('after attn', x2, h_x2)
    cmp('conv raw', conv_mob, h_conv)
    cmp('after conv', x3, h_x3)
    cmp('ff2 raw', ff2_mob, h_ff2)
    cmp('after ff2', x4, h_x4)
    cmp('after norm_out', out_mob, h_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
