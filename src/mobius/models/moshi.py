# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Moshi / PersonaPlex: full-duplex speech-to-speech model.

Architecture (3-model ONNX split):
1. **embedding**: text token + audio codec token fusion
   text_ids (B, S) + audio_codes (B, S, num_codebooks) -> inputs_embeds (B, S, H)
2. **decoder**: causal transformer backbone
   inputs_embeds -> text_logits + standard KV cache
3. **audio_decoder**: depthformer per-codebook autoregressive transformer
   backbone_hidden (B, 1, H) + prev_embedding (B, 1, D) + codebook_idx ->
   codebook_logits (B, 1, audio_logits_size) + depformer KV cache

Unlike LFM2-Audio, Moshi has no mel-spectrogram audio encoder.  Audio is
consumed and produced as RVQ codec token IDs, embedded directly.

HuggingFace weight name prefixes::

    text_emb.           -> embedding.text_emb
    emb.N.              -> embedding.audio_emb.N         (N=0..num_codebooks-1)
    transformer.layers. -> decoder.layers
    out_norm.           -> decoder.out_norm
    text_linear.        -> decoder.lm_head
    depformer_text_emb. -> audio_decoder.depth_text_emb
    depformer_emb.N.    -> audio_decoder.depth_emb.N     (N=0..num_codebooks-2)
    depformer_in.       -> audio_decoder.stacked_depformer_in (stacked)
    depformer.layers.   -> audio_decoder.layers
    linears.            -> audio_decoder.stacked_output_heads (stacked)

Reference: Défossez et al., "Moshi: a speech-text foundation model for
real-time dialogue" (2024). Base model: ``kyutai/moshiko-pytorch-bf16``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig, MoshiConfig
from mobius.components import (
    MLP,
    Attention,
    Embedding,
    Linear,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)

if TYPE_CHECKING:
    import onnx_ir as ir


# ---------------------------------------------------------------------------
# Embedding sub-model
# ---------------------------------------------------------------------------


class _MoshiEmbedding(nn.Module):
    """Moshi embedding: text + audio codec tokens -> inputs_embeds.

    Combines the text token embedding with the sum of all per-codebook audio
    token embeddings.  Each timestep contributes one text token ID and
    ``num_codebooks`` audio codec token IDs.

    Result::

        inputs_embeds = text_emb[text_ids] + sum(audio_emb[i][audio_codes[...,i]])

    Weight names (HF)::

        text_emb.weight        -> text_emb.weight
        emb.N.weight (N=0..K) -> audio_emb.N.weight
    """

    def __init__(self, config: MoshiConfig):
        super().__init__()
        self._num_codebooks = config.num_codebooks
        # Moshi vocabulary has 32001 tokens (32000 text + 1 extra).
        self.text_emb = Embedding(config.vocab_size + 1, config.hidden_size)
        # Per-codebook audio token embeddings: audio_vocab_size x hidden_size each.
        self.audio_emb = nn.ModuleList(
            [
                Embedding(config.audio_vocab_size, config.hidden_size)
                for _ in range(config.num_codebooks)
            ]
        )

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        audio_codes: ir.Value,
    ) -> ir.Value:
        """Forward: (text_ids, audio_codes) -> inputs_embeds.

        Args:
            input_ids:    (batch, seq) int64 text token IDs
            audio_codes:  (batch, seq, num_codebooks) int64 audio codec codes

        Returns:
            inputs_embeds: (batch, seq, hidden_size)
        """
        # Text embedding: (batch, seq, hidden_size)
        embeds = self.text_emb(op, input_ids)

        # Add per-codebook audio embeddings
        for i, emb_table in enumerate(self.audio_emb):
            # Gather codes for codebook i along axis=2: (batch, seq)
            code_i = op.Gather(audio_codes, op.Constant(value_int=i), axis=2)
            embeds = op.Add(embeds, emb_table(op, code_i))

        return embeds


# ---------------------------------------------------------------------------
# Main transformer decoder layers
# ---------------------------------------------------------------------------


class _MoshiDecoderLayer(nn.Module):
    """Single Moshi transformer layer.

    Architecture: PreNorm → Attention → residual → PreNorm → SwiGLU MLP → residual.

    Weight names (HF, under ``transformer.layers.N.``)::

        norm1.alpha [1,1,H]              -> norm1.weight [H]  (via reshape)
        self_attn.in_proj_weight [3H,H]  -> self_attn.{q,k,v}_proj.weight [H,H]
        self_attn.out_proj.weight [H,H]  -> self_attn.o_proj.weight [H,H]
        norm2.alpha [1,1,H]              -> norm2.weight [H]  (via reshape)
        gating.linear_in.weight [2I,H]   -> gating.{gate,up}_proj.weight [I,H]
        gating.linear_out.weight [H,I]   -> gating.down_proj.weight [H,I]
    """

    def __init__(self, config: MoshiConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gating = MLP(config)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        # Pre-norm + causal self-attention
        residual = hidden_states
        hidden_states = self.norm1(op, hidden_states)
        hidden_states, present_kv = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = op.Add(residual, hidden_states)

        # Pre-norm + SwiGLU MLP
        residual = hidden_states
        hidden_states = self.norm2(op, hidden_states)
        hidden_states = self.gating(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_kv


class _MoshiDecoder(nn.Module):
    """Moshi main causal transformer decoder.

    Weight names (HF)::

        transformer.layers.N.* -> layers.N.*  (via preprocess_weights)
        out_norm.alpha [1,1,H] -> out_norm.weight [H]
        text_linear.weight     -> lm_head.weight
    """

    def __init__(self, config: MoshiConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [_MoshiDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.out_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: builder.OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = inputs_embeds
        # RoPE position embeddings
        position_embeddings = self.rotary_emb(op, position_ids)
        # Causal attention bias from the attention mask
        attention_bias = create_attention_bias(op, attention_mask, hidden_states, position_ids)

        present_key_values = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        # Final norm + LM head
        hidden_states = self.out_norm(op, hidden_states)
        logits = self.lm_head(op, hidden_states)

        return logits, present_key_values


# ---------------------------------------------------------------------------
# Depformer audio decoder layers
# ---------------------------------------------------------------------------


class _DepformerLayer(nn.Module):
    """Single Moshi depformer layer.

    Shared causal attention over the codebook-position sequence (KV cache
    accumulates one step per codebook), plus a per-codebook SwiGLU MLP
    selected at runtime by ``codebook_idx``.

    Per-codebook gating weights are stacked as parameters so the correct
    MLP can be selected with a single ``Gather`` on ``codebook_idx``.

    The attention uses ``num_heads = num_codebooks`` and
    ``head_dim = depformer_dim`` (one full-dimensioned head per codebook).

    Weight names (HF, under ``depformer.layers.N.``)::

        norm1.alpha [1,1,D]               -> norm1.weight [D]
        self_attn.in_proj_weight [3*K*D,D] -> self_attn.{q,k,v}_proj.weight [K*D,D]
        self_attn.out_proj.weight [K*D,D] -> self_attn.o_proj.weight [D,K*D] (transposed)
        norm2.alpha [1,1,D]               -> norm2.weight [D]
        gating.M.linear_in.weight [2I,D]  -> stacked_{gate,up}_proj (stacked over M)
        gating.M.linear_out.weight [D,I]  -> stacked_down_proj (stacked over M)
    """

    def __init__(self, config: MoshiConfig):
        super().__init__()
        depformer_dim = config.depformer_dim
        num_codebooks = config.num_codebooks
        interm = config.depformer_intermediate_size

        # Shared attention: num_heads=num_codebooks so each head covers one codebook.
        attn_config = ArchitectureConfig(
            hidden_size=depformer_dim,
            num_attention_heads=num_codebooks,
            num_key_value_heads=num_codebooks,
            head_dim=depformer_dim,  # head_dim = full depformer dim per head
            hidden_act="silu",
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            max_position_embeddings=num_codebooks,
        )
        self.norm1 = RMSNorm(depformer_dim, eps=1e-5)
        self.self_attn = Attention(attn_config)
        self.norm2 = RMSNorm(depformer_dim, eps=1e-5)

        # Per-codebook stacked gating weights:
        # (num_codebooks, intermediate_size, depformer_dim) for gate/up projections.
        # (num_codebooks, depformer_dim, intermediate_size) for down projection.
        self.stacked_gate_proj = nn.Parameter([num_codebooks, interm, depformer_dim])
        self.stacked_up_proj = nn.Parameter([num_codebooks, interm, depformer_dim])
        self.stacked_down_proj = nn.Parameter([num_codebooks, depformer_dim, interm])

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        codebook_idx: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        # Pre-norm + shared causal attention (KV cache holds previous codebook steps)
        residual = hidden_states
        hidden_states = self.norm1(op, hidden_states)
        hidden_states, present_kv = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=None,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = op.Add(residual, hidden_states)

        # Pre-norm + per-codebook SwiGLU MLP (selected by codebook_idx)
        residual = hidden_states
        hidden_states = self.norm2(op, hidden_states)

        # Select gating weights for the current codebook
        # stacked_gate_proj: (num_codebooks, interm, dim) -> (interm, dim)
        gate_w = op.Gather(self.stacked_gate_proj, codebook_idx, axis=0)
        up_w = op.Gather(self.stacked_up_proj, codebook_idx, axis=0)
        down_w = op.Gather(self.stacked_down_proj, codebook_idx, axis=0)

        # SwiGLU: silu(gate_proj(x)) * up_proj(x) -> down_proj
        # gate_w: (interm, dim) -> op.Transpose -> (dim, interm)
        gate = op.MatMul(hidden_states, op.Transpose(gate_w))  # (batch, 1, interm)
        up = op.MatMul(hidden_states, op.Transpose(up_w))  # (batch, 1, interm)
        # SiLU(gate) = gate * sigmoid(gate)
        activated = op.Mul(gate, op.Sigmoid(gate))
        intermediate = op.Mul(activated, up)
        # down_w: (dim, interm) -> op.Transpose -> (interm, dim)
        hidden_states = op.MatMul(intermediate, op.Transpose(down_w))  # (batch, 1, dim)

        hidden_states = op.Add(residual, hidden_states)
        return hidden_states, present_kv


class _MoshiAudioDecoder(nn.Module):
    """Moshi depformer: per-codebook autoregressive depth transformer.

    At each inference step, takes the main transformer's hidden state plus
    the previous codebook's embedding, runs it through the depformer layers,
    and produces logits for the current codebook.

    The depformer KV cache accumulates codebook steps (not time steps).
    Each call advances by one codebook (``codebook_idx`` = 0..num_codebooks-1).

    Weight names (HF)::

        depformer_text_emb.weight  -> depth_text_emb.weight
        depformer_emb.N.weight     -> depth_emb.N.weight  (N=0..num_codebooks-2)
        depformer_in.N.weight      -> stacked_depformer_in (stacked; N=0..num_codebooks-1)
        depformer.layers.N.*       -> layers.N.*  (via preprocess_weights)
        linears.N.weight           -> stacked_output_heads (stacked; N=0..num_codebooks-1)
    """

    def __init__(self, config: MoshiConfig):
        super().__init__()
        depformer_dim = config.depformer_dim
        num_codebooks = config.num_codebooks
        audio_logits_size = config.audio_vocab_size - 1  # 2048: no padding token in output

        # Text depth embedding for codebook-0 input (from main text token)
        self.depth_text_emb = Embedding(config.vocab_size + 1, depformer_dim)
        # Audio depth embeddings for codebooks 1..num_codebooks-1
        self.depth_emb = nn.ModuleList(
            [
                Embedding(config.audio_vocab_size, depformer_dim)
                for _ in range(num_codebooks - 1)
            ]
        )

        # Per-codebook input projections: (num_codebooks, depformer_dim, hidden_size)
        # Gathered at runtime by codebook_idx to project backbone_hidden.
        self.stacked_depformer_in = nn.Parameter(
            [num_codebooks, depformer_dim, config.hidden_size]
        )

        # Depformer layers
        self.layers = nn.ModuleList(
            [_DepformerLayer(config) for _ in range(config.depformer_layers)]
        )

        # Per-codebook output heads: (num_codebooks, audio_logits_size, depformer_dim)
        # Gathered at runtime by codebook_idx to produce logits.
        self.stacked_output_heads = nn.Parameter(
            [num_codebooks, audio_logits_size, depformer_dim]
        )

        # RoPE for depformer (codebook index as position)
        rope_config = ArchitectureConfig(
            hidden_size=depformer_dim,
            num_attention_heads=num_codebooks,
            head_dim=depformer_dim,
            rope_theta=10000.0,
            max_position_embeddings=num_codebooks,
        )
        self.rotary_emb = initialize_rope(rope_config)

        self._num_codebooks = num_codebooks
        self._depformer_dim = depformer_dim

    def forward(
        self,
        op: builder.OpBuilder,
        backbone_hidden: ir.Value,
        prev_embedding: ir.Value,
        codebook_idx: ir.Value,
        past_key_values: list | None = None,
    ):
        """Single-codebook forward pass.

        Args:
            backbone_hidden:  (batch, 1, hidden_size) from main transformer
            prev_embedding:   (batch, 1, depformer_dim) previous codebook embed
            codebook_idx:     scalar int64 — which codebook to predict
            past_key_values:  depformer KV cache (one entry per layer)

        Returns:
            (codebook_logits, present_key_values)
        """
        # Select input projection for current codebook
        # stacked_depformer_in: (num_codebooks, depformer_dim, hidden_size)
        # -> (depformer_dim, hidden_size) after Gather
        in_proj = op.Gather(self.stacked_depformer_in, codebook_idx, axis=0)

        # Project backbone hidden: (batch, 1, hidden_size) @ (hidden_size, depformer_dim)
        depformer_input = op.MatMul(backbone_hidden, op.Transpose(in_proj))  # (batch,1,D)

        # Add previous codebook embedding (provides depth autoregressive context)
        hidden_states = op.Add(depformer_input, prev_embedding)

        # RoPE using codebook_idx as position (shape: [1, 1])
        codebook_pos = op.Reshape(codebook_idx, op.Constant(value_ints=[1, 1]))
        position_embeddings = self.rotary_emb(op, codebook_pos)

        # Run depformer layers with per-codebook MLP selection
        present_key_values = []
        past_kvs = (
            past_key_values if past_key_values is not None else [None] * len(self.layers)
        )
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                codebook_idx=codebook_idx,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        # Select output head for current codebook
        # stacked_output_heads: (num_codebooks, audio_logits_size, depformer_dim)
        # -> (audio_logits_size, depformer_dim) after Gather
        head_w = op.Gather(self.stacked_output_heads, codebook_idx, axis=0)

        # Logits: (batch, 1, depformer_dim) @ (depformer_dim, audio_logits_size)
        codebook_logits = op.MatMul(hidden_states, op.Transpose(head_w))  # (batch,1,V)

        return codebook_logits, present_key_values


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class MoshiModel(nn.Module):
    """Moshi/PersonaPlex audio-to-audio model (3-model ONNX split).

    Used with ``MoshiTask`` which builds:
    - ``embedding``:    text + audio codec token embeddings
    - ``decoder``:      32-layer causal transformer, text logits + KV cache
    - ``audio_decoder``: 6-layer depformer, per-codebook audio logits

    Weight names are mapped from the Moshi HuggingFace checkpoint format
    via ``preprocess_weights``.
    """

    default_task: str = "moshi"
    config_class: type = MoshiConfig

    def __init__(self, config: MoshiConfig):
        super().__init__()
        self.embedding = _MoshiEmbedding(config)
        self.decoder = _MoshiDecoder(config)
        self.audio_decoder = _MoshiAudioDecoder(config)
        self._config = config

    @staticmethod
    def preprocess_weights(state_dict: dict) -> dict:
        """Map Moshi HF checkpoint weights to mobius module names.

        Transforms:
        - ``norm*.alpha [1,1,H]`` → ``norm*.weight [H]`` (squeeze)
        - ``self_attn.in_proj_weight [3H,H]`` → split into ``q/k/v_proj.weight``
        - ``self_attn.out_proj.weight`` → ``self_attn.o_proj.weight``
        - ``gating.linear_in.weight [2I,H]`` → split into ``gate/up_proj.weight``
        - ``gating.linear_out.weight`` → ``gating.down_proj.weight``
        - ``depformer_in.N.weight`` → stacked ``stacked_depformer_in``
        - ``linears.N.weight`` → stacked ``stacked_output_heads``
        - Per-codebook gating → stacked ``stacked_{gate,up,down}_proj``
        """
        new_sd: dict = {}

        # Count transformer and depformer layers
        num_transformer_layers = sum(
            1
            for k in state_dict
            if k.startswith("transformer.layers.") and k.endswith(".norm1.alpha")
        )
        num_depformer_layers = sum(
            1
            for k in state_dict
            if k.startswith("depformer.layers.") and k.endswith(".norm1.alpha")
        )

        # Count codebooks from emb.* keys
        num_codebooks = sum(
            1 for k in state_dict if k.startswith("emb.") and k.endswith(".weight")
        )

        # ----------------------------------------------------------------
        # Embedding sub-model
        # ----------------------------------------------------------------
        if "text_emb.weight" in state_dict:
            new_sd["embedding.text_emb.weight"] = state_dict.pop("text_emb.weight")
        for i in range(num_codebooks):
            key = f"emb.{i}.weight"
            if key in state_dict:
                new_sd[f"embedding.audio_emb.{i}.weight"] = state_dict.pop(key)

        # ----------------------------------------------------------------
        # Main transformer decoder layers
        # ----------------------------------------------------------------
        for i in range(num_transformer_layers):
            prefix = f"transformer.layers.{i}"
            dst = f"decoder.layers.{i}"

            # RMSNorm: alpha [1,1,H] → weight [H]
            for norm in ("norm1", "norm2"):
                alpha_key = f"{prefix}.{norm}.alpha"
                if alpha_key in state_dict:
                    new_sd[f"{dst}.{norm}.weight"] = state_dict.pop(alpha_key).flatten()

            # Attention: split packed QKV in_proj_weight
            in_proj_key = f"{prefix}.self_attn.in_proj_weight"
            if in_proj_key in state_dict:
                q, k, v = state_dict.pop(in_proj_key).chunk(3, dim=0)
                new_sd[f"{dst}.self_attn.q_proj.weight"] = q
                new_sd[f"{dst}.self_attn.k_proj.weight"] = k
                new_sd[f"{dst}.self_attn.v_proj.weight"] = v

            # out_proj -> o_proj (square matrix, no transpose needed for hidden_size=head_dim*heads)
            out_proj_key = f"{prefix}.self_attn.out_proj.weight"
            if out_proj_key in state_dict:
                new_sd[f"{dst}.self_attn.o_proj.weight"] = state_dict.pop(out_proj_key)

            # Gating MLP: split linear_in, rename linear_out
            lin_in_key = f"{prefix}.gating.linear_in.weight"
            if lin_in_key in state_dict:
                gate, up = state_dict.pop(lin_in_key).chunk(2, dim=0)
                new_sd[f"{dst}.gating.gate_proj.weight"] = gate
                new_sd[f"{dst}.gating.up_proj.weight"] = up

            lin_out_key = f"{prefix}.gating.linear_out.weight"
            if lin_out_key in state_dict:
                new_sd[f"{dst}.gating.down_proj.weight"] = state_dict.pop(lin_out_key)

        # Decoder final norm and LM head
        if "out_norm.alpha" in state_dict:
            new_sd["decoder.out_norm.weight"] = state_dict.pop("out_norm.alpha").flatten()
        if "text_linear.weight" in state_dict:
            new_sd["decoder.lm_head.weight"] = state_dict.pop("text_linear.weight")

        # ----------------------------------------------------------------
        # Audio decoder (depformer)
        # ----------------------------------------------------------------

        # Text depth embedding
        if "depformer_text_emb.weight" in state_dict:
            new_sd["audio_decoder.depth_text_emb.weight"] = state_dict.pop(
                "depformer_text_emb.weight"
            )

        # Per-codebook depth audio embeddings (codebooks 1..num_codebooks-1)
        for i in range(num_codebooks - 1):
            key = f"depformer_emb.{i}.weight"
            if key in state_dict:
                new_sd[f"audio_decoder.depth_emb.{i}.weight"] = state_dict.pop(key)

        # Stack per-codebook input projections: (num_codebooks, depformer_dim, hidden_size)
        in_projs = []
        for i in range(num_codebooks):
            key = f"depformer_in.{i}.weight"
            if key in state_dict:
                in_projs.append(state_dict.pop(key))
        if in_projs:
            new_sd["audio_decoder.stacked_depformer_in"] = torch.stack(in_projs, dim=0)

        # Stack output heads: (num_codebooks, audio_logits_size, depformer_dim)
        out_heads = []
        for i in range(num_codebooks):
            key = f"linears.{i}.weight"
            if key in state_dict:
                out_heads.append(state_dict.pop(key))
        if out_heads:
            new_sd["audio_decoder.stacked_output_heads"] = torch.stack(out_heads, dim=0)

        # Depformer layers
        for i in range(num_depformer_layers):
            dep_prefix = f"depformer.layers.{i}"
            dst = f"audio_decoder.layers.{i}"

            # RMSNorm
            for norm in ("norm1", "norm2"):
                alpha_key = f"{dep_prefix}.{norm}.alpha"
                if alpha_key in state_dict:
                    new_sd[f"{dst}.{norm}.weight"] = state_dict.pop(alpha_key).flatten()

            # Attention: split packed QKV (3 * num_codebooks * depformer_dim, depformer_dim)
            in_proj_key = f"{dep_prefix}.self_attn.in_proj_weight"
            if in_proj_key in state_dict:
                q, k, v = state_dict.pop(in_proj_key).chunk(3, dim=0)
                new_sd[f"{dst}.self_attn.q_proj.weight"] = q
                new_sd[f"{dst}.self_attn.k_proj.weight"] = k
                new_sd[f"{dst}.self_attn.v_proj.weight"] = v

            # out_proj is stored transposed: [K*D, D] in HF vs [D, K*D] in mobius.
            out_proj_key = f"{dep_prefix}.self_attn.out_proj.weight"
            if out_proj_key in state_dict:
                # Transpose from [K*D, D] to [D, K*D] for Linear(K*D, D).weight = [D, K*D]
                new_sd[f"{dst}.self_attn.o_proj.weight"] = state_dict.pop(out_proj_key).T

            # Per-codebook gating: stack into stacked_{gate,up,down}_proj
            gate_projs, up_projs, down_projs = [], [], []
            for j in range(num_codebooks):
                lin_in_key = f"{dep_prefix}.gating.{j}.linear_in.weight"
                if lin_in_key in state_dict:
                    gate, up = state_dict.pop(lin_in_key).chunk(2, dim=0)
                    gate_projs.append(gate)
                    up_projs.append(up)

                lin_out_key = f"{dep_prefix}.gating.{j}.linear_out.weight"
                if lin_out_key in state_dict:
                    down_projs.append(state_dict.pop(lin_out_key))

            if gate_projs:
                new_sd[f"{dst}.stacked_gate_proj"] = torch.stack(gate_projs, dim=0)
                new_sd[f"{dst}.stacked_up_proj"] = torch.stack(up_projs, dim=0)
            if down_projs:
                new_sd[f"{dst}.stacked_down_proj"] = torch.stack(down_projs, dim=0)

        # Pass through any remaining weights unchanged
        new_sd.update(state_dict)
        return new_sd
