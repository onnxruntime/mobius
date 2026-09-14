# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Moshi full-duplex speech-to-speech language model (Kyutai / NVIDIA).

This module builds the ONNX graphs for the Moshi LM used by
``nvidia/personaplex-7b-v1``.  The Moshi LM has two transformer stacks:

* **Temporal transformer** (:class:`MoshiTemporalModel`): a 7B decoder-only
  transformer over a 17-channel token frame (channel 0 = text, channels
  1..16 = audio codebooks).  Per step it embeds and sums the 17 channels,
  runs 32 RMSNorm/SwiGLU layers with sliding-window causal attention, and
  emits the post-norm hidden state plus text logits.

* **Depformer** (:class:`MoshiDepformerModel`): a small 6-layer transformer
  that autoregressively predicts one frame of audio codebooks. Public
  Moshi/Moshiko checkpoints use 8 substeps; PersonaPlex uses 16. This module
  builds both variants and maps their native per-step weights.

The checkpoint is the native Kyutai ``safetensors`` format (no HuggingFace
``config.json``); :func:`_moshi_temporal_config` reproduces the fixed
dimensions and :func:`_preprocess_moshi_temporal_weights` maps the native
parameter names onto the mobius component tree.

Reference: ``moshi.models.lm.LMModel`` (Kyutai), ``modules/transformer.py``,
``modules/gating.py`` in the personaplex release.
"""

from __future__ import annotations

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import (
    Embedding,
    FusedGateUpMLP,
    Linear,
    RMSNorm,
)
from mobius.models.base import TextModel
from mobius.models.mimi import _interleaved_to_halfsplit

# --- Temporal transformer dimensions (personaplex-7b-v1) ---
_T_DIM = 4096
_T_LAYERS = 32
_T_HEADS = 32
_T_HEAD_DIM = 128
_T_FFN = 11264  # SwiGLU hidden = (2 * int(4.125 * 4096)) // 3
_T_CONTEXT = 3000  # sliding-window attention context
_T_RMS_EPS = 1e-8
_ROPE_THETA = 10000.0

# --- Token vocabularies ---
_NUM_CH = 17  # 1 text + 16 audio codebooks
_NUM_AUDIO_CB = 16
_AUDIO_CARD = 2048  # audio codebook size; embedding table has card + 1 rows
_TEXT_CARD = 32000  # text vocabulary; embedding table has card + 1 rows
_TEXT_VOCAB = 32000  # text_linear output (logits) size

# --- Depformer dimensions ---
_D_DIM = 1024
_D_LAYERS = 6
_D_HEADS = 16
_D_HEAD_DIM = 64
_D_GATE_HIDDEN = 2816  # SwiGLU hidden = (2 * int(4.125 * 1024)) // 3
_D_Q = 16  # PersonaPlex predicts 16 codebooks; public Moshi predicts 8.
_D_RMS_EPS = 1e-8


class _MoshiTemporalEmbedding(nn.Module):
    """Sum the 17-channel token frame into a single hidden embedding.

    Mirrors ``LMModel.embed_codes``: ``text_emb(frame[:, 0])`` plus
    ``emb[cb](frame[:, cb + 1])`` for ``cb`` in ``range(16)``. Channel 0 is the
    text stream; channels 1..16 are the audio codebooks (``audio_offset = 1``).
    """

    def __init__(self):
        super().__init__()
        # Audio embeddings: card + 1 rows (extra row = initial/zero token).
        self.emb = nn.ModuleList(
            [Embedding(_AUDIO_CARD + 1, _T_DIM) for _ in range(_NUM_AUDIO_CB)]
        )
        self.text_emb = Embedding(_TEXT_CARD + 1, _T_DIM)

    def forward(self, op: OpBuilder, frame: ir.Value) -> ir.Value:
        # frame: (B, 17, S) INT64. Gather each channel -> (B, S) ids.
        text_ids = op.Gather(frame, 0, axis=1)  # (B, S)
        hidden = self.text_emb(op, text_ids)  # (B, S, 4096)
        for cb in range(_NUM_AUDIO_CB):
            audio_ids = op.Gather(frame, cb + 1, axis=1)  # (B, S)
            hidden = op.Add(hidden, self.emb[cb](op, audio_ids))
        return hidden


class MoshiTemporalModel(nn.Module):
    """Moshi temporal (7B) transformer: token frame -> hidden + text logits.

    Outputs:
        - ``hidden``: (batch, seq_len, 4096) post-``out_norm`` transformer state,
          consumed by the depformer.
        - ``text_logits``: (batch, seq_len, 32000) text-stream logits.
        - present key/value cache for each of the 32 layers.

    Replicates ``LMModel.forward_codes`` / ``forward_embeddings`` (Kyutai).
    """

    default_task: str = "moshi-temporal"
    category: str = "Audio"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.embed = _MoshiTemporalEmbedding()
        # Reuse the standard decoder stack: RMSNorm + RoPE + SwiGLU (silu).
        # The fused SwiGLU weight (linear_in) maps onto FusedGateUpMLP's
        # gate_up_proj, so no MLP weight splitting is needed.
        self.model = TextModel(config, mlp_class=FusedGateUpMLP)
        # out_norm (LMModel.out_norm) is TextModel.norm; text_linear is the
        # text-stream LM head (no bias).
        self.text_linear = Linear(_T_DIM, _TEXT_VOCAB, bias=False)

    def forward(
        self,
        op: OpBuilder,
        input_frame: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        inputs_embeds = self.embed(op, input_frame)  # (B, S, 4096)
        hidden, present_key_values = self.model(
            op,
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        text_logits = self.text_linear(op, hidden)  # (B, S, 32000)
        return hidden, text_logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return _preprocess_moshi_temporal_weights(state_dict)


def _moshi_temporal_config() -> ArchitectureConfig:
    """Fixed :class:`ArchitectureConfig` for the Moshi temporal transformer.

    The native Kyutai checkpoint ships no ``config.json``; all dimensions are
    architectural constants. RMSNorm uses ``eps = 1e-8`` and RoPE ``theta =
    1e4`` (Kyutai ``rms_norm_f32`` / ``rope``).
    """
    return ArchitectureConfig(
        model_type="moshi",
        hidden_size=_T_DIM,
        num_hidden_layers=_T_LAYERS,
        num_attention_heads=_T_HEADS,
        num_key_value_heads=_T_HEADS,
        head_dim=_T_HEAD_DIM,
        intermediate_size=_T_FFN,
        vocab_size=_TEXT_VOCAB,
        max_position_embeddings=8000,
        rms_norm_eps=_T_RMS_EPS,
        rope_theta=_ROPE_THETA,
        rope_type="default",
        hidden_act="silu",
        sliding_window=_T_CONTEXT,
        dtype=ir.DataType.FLOAT,
    )


# ---------------------------------------------------------------------------
# Weight preprocessing: native Kyutai Moshi LM -> mobius module parameter names.
# ---------------------------------------------------------------------------


def _preprocess_moshi_temporal_weights(
    sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map native Kyutai temporal-transformer weights onto mobius names.

    * ``text_emb`` / ``emb.{i}`` -> ``embed.text_emb`` / ``embed.emb.{i}``;
    * fused ``self_attn.in_proj_weight`` is split into q/k/v and Q/K rows are
      permuted from interleaved-pair RoPE to mobius half-split RoPE;
    * RMSNorm ``alpha`` tensors (shape ``(1, 1, dim)``) are squeezed to
      ``(dim,)`` and renamed to ``input_layernorm`` / ``post_attention_layernorm``
      / ``norm`` weights;
    * fused SwiGLU ``gating.linear_in`` / ``linear_out`` map directly onto
      ``mlp.gate_up_proj`` / ``mlp.down_proj``.
    """
    out: dict[str, torch.Tensor] = {}

    # Embeddings.
    out["embed.text_emb.weight"] = sd["text_emb.weight"]
    for cb in range(_NUM_AUDIO_CB):
        out[f"embed.emb.{cb}.weight"] = sd[f"emb.{cb}.weight"]

    # Transformer layers.
    for i in range(_T_LAYERS):
        src = f"transformer.layers.{i}"
        dst = f"model.layers.{i}"

        # RMSNorm alpha (1, 1, D) -> weight (D,).
        out[f"{dst}.input_layernorm.weight"] = sd[f"{src}.norm1.alpha"].reshape(-1)
        out[f"{dst}.post_attention_layernorm.weight"] = sd[f"{src}.norm2.alpha"].reshape(-1)

        # Fused QKV -> q/k/v; permute q,k for the RoPE convention.
        in_proj = sd[f"{src}.self_attn.in_proj_weight"]  # (3*D, D)
        dim = in_proj.shape[1]
        q, k, v = in_proj[:dim], in_proj[dim : 2 * dim], in_proj[2 * dim :]
        out[f"{dst}.self_attn.q_proj.weight"] = _interleaved_to_halfsplit(q, _T_HEAD_DIM)
        out[f"{dst}.self_attn.k_proj.weight"] = _interleaved_to_halfsplit(k, _T_HEAD_DIM)
        out[f"{dst}.self_attn.v_proj.weight"] = v
        out[f"{dst}.self_attn.o_proj.weight"] = sd[f"{src}.self_attn.out_proj.weight"]

        # Fused SwiGLU gating: linear_in == [gate; up] stacked, linear_out == down.
        out[f"{dst}.mlp.gate_up_proj.weight"] = sd[f"{src}.gating.linear_in.weight"]
        out[f"{dst}.mlp.down_proj.weight"] = sd[f"{src}.gating.linear_out.weight"]

    # Final norm (out_norm) and text LM head.
    out["model.norm.weight"] = sd["out_norm.alpha"].reshape(-1)
    out["text_linear.weight"] = sd["text_linear.weight"]

    return out


# ===========================================================================
# Depformer: autoregressive per-substep audio codebook predictor.
# ===========================================================================
#
# The depformer predicts one frame of audio codebooks, one substep at a time.
# It is exported as a ONE-SUBSTEP graph: given the
# temporal hidden state, the previously sampled token, the substep index, and
# the intra-frame KV cache, it produces the logits for one codebook and the
# updated KV cache.  The deployment loop (see the ORT example) calls this graph
# 8 times for Moshi/Moshiko or 16 times for PersonaPlex, feeding each substep
# the token sampled previously; the cache resets at each temporal frame.
#
# ``weights_per_step``: the attention in/out projections, the SwiGLU gating,
# the per-codebook input projection, and the output head all use a different
# weight slice for each of the 8 or 16 substeps, selected by ``substep_index`` via
# a runtime Gather over a stacked weight tensor.  The two RMSNorms are shared
# across substeps.  The depformer uses NO positional embedding (no RoPE) and
# full causal attention over the cached substeps.


class _PerStepLinear(nn.Module):
    """Bias-free linear whose weight is selected per substep.

    The weight is stored stacked as ``(num_steps, out_features, in_features)``;
    ``forward`` gathers the ``substep_index`` slice and applies it. Mirrors
    Kyutai ``multi_linear`` (``modules/transformer.py``).
    """

    def __init__(self, num_steps: int, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter([num_steps, out_features, in_features])

    def forward(self, op: OpBuilder, x: ir.Value, substep_index: ir.Value) -> ir.Value:
        w = op.Gather(self.weight, substep_index, axis=0)  # (out, in)
        w_t = op.Transpose(w, perm=[1, 0])  # (in, out)
        return op.MatMul(x, w_t)


class _DepformerLayer(nn.Module):
    """One depformer transformer layer (per-step attention + gating)."""

    def __init__(self, dep_q: int):
        super().__init__()
        self.norm1 = RMSNorm(_D_DIM, eps=_D_RMS_EPS)
        # Fused QKV per step: out = 3 * dim.
        self.in_proj = _PerStepLinear(dep_q, _D_DIM, 3 * _D_DIM)
        self.out_proj = _PerStepLinear(dep_q, _D_DIM, _D_DIM)
        self.norm2 = RMSNorm(_D_DIM, eps=_D_RMS_EPS)
        # SwiGLU gating per step: linear_in -> 2 * hidden, linear_out -> dim.
        self.gating_in = _PerStepLinear(dep_q, _D_DIM, 2 * _D_GATE_HIDDEN)
        self.gating_out = _PerStepLinear(dep_q, _D_GATE_HIDDEN, _D_DIM)
        self._scale = 1.0 / (_D_HEAD_DIM**0.5)

    def forward(
        self,
        op: OpBuilder,
        x: ir.Value,
        substep_index: ir.Value,
        past_key: ir.Value,
        past_value: ir.Value,
    ):
        # x: (B, 1, 1024). Pre-norm self-attention.
        h = self.norm1(op, x)
        qkv = self.in_proj(op, h, substep_index)  # (B, 1, 3 * 1024)
        q, k, v = op.Split(qkv, axis=-1, num_outputs=3, _outputs=3)
        # Full causal attention over the cached substeps (no RoPE). q_len == 1,
        # so the query attends to every cached key/value (is_causal=1).
        attn, present_key, present_value = op.Attention(
            q,
            k,
            v,
            None,  # no attn mask
            past_key,
            past_value,
            q_num_heads=_D_HEADS,
            kv_num_heads=_D_HEADS,
            scale=self._scale,
            is_causal=1,
            _outputs=3,
        )
        attn = self.out_proj(op, attn, substep_index)
        x = op.Add(x, attn)  # residual (no layer_scale)

        # Pre-norm SwiGLU gating.
        h2 = self.norm2(op, x)
        gate_up = self.gating_in(op, h2, substep_index)  # (B, 1, 2 * hidden)
        gate, up = op.Split(gate_up, axis=-1, num_outputs=2, _outputs=2)
        ff = op.Mul(op.Swish(gate), up)  # silu(gate) * up
        ff = self.gating_out(op, ff, substep_index)  # (B, 1, 1024)
        x = op.Add(x, ff)  # residual (no layer_scale)
        return x, present_key, present_value


class MoshiDepformerModel(nn.Module):
    """Moshi depformer: one autoregressive substep of audio-codebook prediction.

    Inputs (per substep):
        - ``hidden``: (batch, 1, 4096) temporal transformer output for this
          frame.
        - ``prev_token``: (batch, 1) INT64 token sampled at the previous
          substep (the temporal text token for substep 0).
        - ``substep_index``: scalar INT64 in ``[0, dep_q - 1]`` selecting the
          per-step weights and the input-embedding table.
        - per-layer KV cache (6 layers).

    Output:
        - ``logits``: (batch, 1, 2048) logits for audio codebook
          ``substep_index``.
        - updated per-layer KV cache.

    Replicates ``LMModel.forward_depformer`` (Kyutai).
    """

    default_task: str = "moshi-depformer"
    category: str = "Audio"

    def __init__(self, config: ArchitectureConfig | None = None):
        super().__init__()
        self.config = config
        self.dep_q = config.max_position_embeddings if config is not None else _D_Q
        if self.dep_q not in (8, 16):
            raise ValueError(
                f"dep_q must be 8 (Moshi/Moshiko) or 16 (PersonaPlex), got {self.dep_q}"
            )
        # Per-codebook input projection (temporal dim -> depformer dim).
        self.depformer_in = _PerStepLinear(self.dep_q, _T_DIM, _D_DIM)
        # Previous-token embeddings: text table (substep 0) + audio tables
        # (substeps 1..dep_q-1 use audio table substep-1).
        self.text_emb = Embedding(_TEXT_CARD + 1, _D_DIM)
        self.audio_emb = nn.Parameter([self.dep_q - 1, _AUDIO_CARD + 1, _D_DIM])
        self.layers = nn.ModuleList([_DepformerLayer(self.dep_q) for _ in range(_D_LAYERS)])
        # Output heads (depformer dim -> audio codebook logits).
        self.linears = _PerStepLinear(self.dep_q, _D_DIM, _AUDIO_CARD)

    def _embed_prev(
        self, op: OpBuilder, prev_token: ir.Value, substep_index: ir.Value
    ) -> ir.Value:
        # Select text embedding (substep 0) vs audio embedding (substep >= 1)
        # without an If: compute both with OOB-guarded indices, then Where.
        is_first = op.Equal(substep_index, 0)  # scalar bool
        zero = op.Constant(value_int=0)  # scalar INT64
        # Audio table index = substep - 1, clamped to a valid row for substep 0.
        audio_tbl_idx = op.Max(op.Sub(substep_index, 1), zero)
        # Guard token ids so the unused branch never indexes out of bounds
        # (text ids can exceed the audio table size and vice versa).
        text_token = op.Where(is_first, prev_token, op.CastLike(zero, prev_token))
        audio_token = op.Where(is_first, op.CastLike(zero, prev_token), prev_token)
        text_e = self.text_emb(op, text_token)  # (B, 1, 1024)
        audio_tbl = op.Gather(self.audio_emb, audio_tbl_idx, axis=0)  # (2049, 1024)
        audio_e = op.Gather(audio_tbl, audio_token)  # (B, 1, 1024)
        return op.Where(is_first, text_e, audio_e)

    def forward(
        self,
        op: OpBuilder,
        hidden: ir.Value,
        prev_token: ir.Value,
        substep_index: ir.Value,
        past_key_values: list | None = None,
    ):
        proj = self.depformer_in(op, hidden, substep_index)  # (B, 1, 1024)
        emb = self._embed_prev(op, prev_token, substep_index)  # (B, 1, 1024)
        x = op.Add(proj, emb)

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            past_key = past_kv[0] if past_kv is not None else None
            past_value = past_kv[1] if past_kv is not None else None
            x, present_key, present_value = layer(op, x, substep_index, past_key, past_value)
            present_key_values.append((present_key, present_value))

        logits = self.linears(op, x, substep_index)  # (B, 1, 2048)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return _preprocess_moshi_depformer_weights(state_dict, dep_q=self.dep_q)


def _moshi_depformer_config(dep_q: int = _D_Q) -> ArchitectureConfig:
    """Return the fixed-size Moshi (8) or PersonaPlex (16) depformer config."""
    if dep_q not in (8, 16):
        raise ValueError(f"dep_q must be 8 (Moshi/Moshiko) or 16 (PersonaPlex), got {dep_q}")
    return ArchitectureConfig(
        model_type="moshi_depformer",
        hidden_size=_D_DIM,
        num_hidden_layers=_D_LAYERS,
        num_attention_heads=_D_HEADS,
        num_key_value_heads=_D_HEADS,
        head_dim=_D_HEAD_DIM,
        intermediate_size=_D_GATE_HIDDEN,
        vocab_size=_AUDIO_CARD,
        max_position_embeddings=dep_q,
        rms_norm_eps=_D_RMS_EPS,
        rope_type=None,
        hidden_act="silu",
        dtype=ir.DataType.FLOAT,
    )


def _preprocess_moshi_depformer_weights(
    sd: dict[str, torch.Tensor],
    *,
    dep_q: int = _D_Q,
) -> dict[str, torch.Tensor]:
    """Map native Kyutai depformer weights onto mobius parameter names.

    Per-step weights stored stacked as ``(steps * out, in)`` (or
    ``(steps * 3 * dim, in)`` for fused QKV) are reshaped to
    ``(steps, out, in)``; the shared RMSNorm ``alpha`` tensors are squeezed.
    """
    if dep_q not in (8, 16):
        raise ValueError(f"dep_q must be 8 (Moshi/Moshiko) or 16 (PersonaPlex), got {dep_q}")
    out: dict[str, torch.Tensor] = {}

    # Per-codebook input projections: depformer_in.{i}.weight (1024, 4096).
    in_proj = torch.stack(
        [sd[f"depformer_in.{i}.weight"] for i in range(dep_q)], dim=0
    )  # (dep_q, 1024, 4096)
    out["depformer_in.weight"] = in_proj

    # Previous-token embeddings.
    out["text_emb.weight"] = sd["depformer_text_emb.weight"]
    out["audio_emb"] = torch.stack(
        [sd[f"depformer_emb.{i}.weight"] for i in range(dep_q - 1)], dim=0
    )  # (dep_q - 1, 2049, 1024)

    # Output heads: linears.{i}.weight (2048, 1024).
    out["linears.weight"] = torch.stack(
        [sd[f"linears.{i}.weight"] for i in range(dep_q)], dim=0
    )  # (dep_q, 2048, 1024)

    # Transformer layers.
    for i in range(_D_LAYERS):
        src = f"depformer.layers.{i}"
        dst = f"layers.{i}"
        # Shared RMSNorm alpha (1, 1, D) -> (D,).
        out[f"{dst}.norm1.weight"] = sd[f"{src}.norm1.alpha"].reshape(-1)
        out[f"{dst}.norm2.weight"] = sd[f"{src}.norm2.alpha"].reshape(-1)
        # Per-step fused QKV: (dep_q * 3 * 1024, 1024)
        # -> (dep_q, 3 * 1024, 1024).
        in_w = sd[f"{src}.self_attn.in_proj_weight"]
        out[f"{dst}.in_proj.weight"] = in_w.reshape(dep_q, 3 * _D_DIM, _D_DIM)
        # Per-step output projection: (dep_q * 1024, 1024)
        # -> (dep_q, 1024, 1024).
        out_w = sd[f"{src}.self_attn.out_proj.weight"]
        out[f"{dst}.out_proj.weight"] = out_w.reshape(dep_q, _D_DIM, _D_DIM)
        # Per-step SwiGLU gating.
        gin = torch.stack(
            [sd[f"{src}.gating.{t}.linear_in.weight"] for t in range(dep_q)], dim=0
        )  # (dep_q, 2 * hidden, 1024)
        gout = torch.stack(
            [sd[f"{src}.gating.{t}.linear_out.weight"] for t in range(dep_q)], dim=0
        )  # (dep_q, 1024, hidden)
        out[f"{dst}.gating_in.weight"] = gin
        out[f"{dst}.gating_out.weight"] = gout

    return out
