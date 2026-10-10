"""HF hook setup and RoPE alignment for the layer-0 Qwen3 attention probe."""

from __future__ import annotations

import torch

from tensorrt_debug.comparison import compare_tensor


class AttentionReference:
    """Capture HF intermediates; leave the mask experiment in inspect_attention.py."""

    def __init__(self, runner, reference_model):
        if not {"probe.query", "probe.attention"}.issubset(runner.names):
            raise ValueError("Use an engine built by tensorrt_attention_probe.py")
        self.runner = runner
        self.model = reference_model
        self.attention = reference_model.model.layers[0].self_attn
        self.intermediates = {}
        self.hooks = [
            self.attention.o_proj.register_forward_pre_hook(self._capture_attention),
            self.attention.q_norm.register_forward_hook(self._capture_query),
        ]

    def _capture_attention(self, module, inputs):
        self.intermediates["attention"] = inputs[0].detach().float().cpu().numpy().copy()

    def _capture_query(self, module, inputs, output):
        self.intermediates["query"] = output.detach().clone()

    def compare_with_hf(self, feeds, actual_query, actual_attention):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        with torch.inference_mode():
            # HF captures Q before RoPE; the diagnostic engine exposes it after RoPE.
            query = self.intermediates["query"].transpose(1, 2)
            cos, sin = self.model.model.rotary_emb(
                query, torch.from_numpy(feeds["position_ids"].copy())
            )
            query, _ = apply_rotary_pos_emb(query, query, cos, sin)
        compare_tensor(
            "Layer-0 post-RoPE query vs HF", actual_query, query.float().cpu().numpy()
        )
        compare_tensor(
            "Layer-0 pre-projection attention vs HF",
            actual_attention,
            self.intermediates["attention"],
        )

    def close(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
