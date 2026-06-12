# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Phi model variants: Phi-1/2, Phi-4MM (multimodal), Phi-3 Small.

Replicates HuggingFace's ``PhiForCausalLM``, ``Phi4MMForCausalLM``,
and ``Phi3SmallForCausalLM``. Phi-4MM adds LoRA adapters for vision
and audio modalities. Phi-3 Small uses block-sparse attention with
MuP (maximal update parameterization) scaling.
"""

from __future__ import annotations

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import rename_weight_keys, split_fused_qkv
from mobius.components import (
    FCMLP,
    ConformerEncoder,
    Embedding,
    FusedGateUpMLP,
    InputMixer,
    LayerNorm,
    Linear,
    PatchEmbedding,
    RMSNorm,
    VisionEncoder,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._attention import Attention
from mobius.components._decoder import DecoderLayer
from mobius.components._lora import LoRALinear
from mobius.models.base import CausalLMModel, TextModel
from mobius.models.phi3 import Phi3CausalLMModel


class _PhiDecoderLayer(nn.Module):
    """Phi-1/2 decoder layer with single-norm parallel residual.

    A single LayerNorm is applied to the hidden states, then both attention
    and MLP receive the same normalized output. Their results are summed
    with the residual in a single addition:

        ln_out = input_layernorm(hidden)
        out = hidden + attn(ln_out) + mlp(ln_out)

    This is the same pattern as GPT-J/CodeGen. Unlike Phi-3 which is
    sequential (with a separate ``post_attention_layernorm`` before the MLP),
    Phi-1/2 shares one norm between both branches.

    Attribute names match HF ``PhiDecoderLayer``:
    - ``input_layernorm`` for the shared LayerNorm
    - ``self_attn`` for the attention module
    - ``mlp`` for the FCMLP module
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        # Single shared norm (no post_attention_layernorm in Phi-1/2)
        self.input_layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config)  # 'self_attn' matches HF attribute name
        self.mlp = FCMLP(
            config.hidden_size,
            config.intermediate_size,
            activation=config.hidden_act or "gelu",
            bias=config.mlp_bias,
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
    ) -> tuple[ir.Value, tuple]:
        residual = hidden_states

        # Single norm shared between attention and MLP
        ln_out = self.input_layernorm(op, hidden_states)  # (B, S, H)
        attn_out, present_kv = self.self_attn(
            op, ln_out, attention_bias, position_embeddings, past_key_value
        )
        mlp_out = self.mlp(op, ln_out)  # same ln_out as attention

        # Parallel residual: both branches added in one step
        hidden_states = op.Add(residual, op.Add(attn_out, mlp_out))
        return hidden_states, present_kv


class _PhiTextModel(nn.Module):
    """Phi-1/2 backbone with RoPE and full LayerNorm.

    Attribute names match HF ``PhiModel``:
    - ``embed_tokens`` for the token embedding
    - ``layers`` for the decoder layer list
    - ``final_layernorm`` for the output norm (HF uses this name, not ``norm``)
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [_PhiDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        # HF Phi names the final norm "final_layernorm" (not "norm" as in Llama)
        self.final_layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ) -> tuple[ir.Value, list]:
        hidden_states = self.embed_tokens(op, input_ids)
        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op, hidden_states, attention_bias, position_embeddings, past_kv
            )
            present_key_values.append(present_kv)

        hidden_states = self.final_layernorm(op, hidden_states)
        return hidden_states, present_key_values


class PhiCausalLMModel(CausalLMModel):
    """Phi-1/2 causal language model with parallel attention.

    Differences from the Llama-style ``CausalLMModel``:
    - Single-norm parallel residual (like GPT-J, not sequential like Llama)
    - Non-gated FCMLP instead of gated MLP
    - Full LayerNorm throughout (not RMSNorm)
    - LM head has a bias term (``lm_head.bias``)
    - Final norm is ``model.final_layernorm`` (matches HF attribute)

    Replicates HuggingFace's ``PhiForCausalLM``.
    """

    default_task: str = "text-generation"
    category: str = "Text Generation"

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model = _PhiTextModel(config)
        # Phi LM head has a bias term
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=True)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op, input_ids, attention_mask, position_ids, past_key_values
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HF Phi weight names to our ONNX attribute names.

        Most paths match directly (model.embed_tokens, model.layers.N.*,
        model.final_layernorm, lm_head, self_attn.q/k/v_proj). Three renames:

        1. Output proj: ``self_attn.dense.*`` → ``self_attn.o_proj.*``
        2. MLP up:   ``mlp.fc1.*`` → ``mlp.up_proj.*``
        3. MLP down: ``mlp.fc2.*`` → ``mlp.down_proj.*``
        """
        new_state_dict: dict[str, torch.Tensor] = rename_weight_keys(
            state_dict,
            [
                (".self_attn.dense.", ".self_attn.o_proj."),
                (".mlp.fc1.", ".mlp.up_proj."),
                (".mlp.fc2.", ".mlp.down_proj."),
            ],
        )
        return super().preprocess_weights(new_state_dict)


def _parse_lora_adapters(
    config: ArchitectureConfig,
) -> list[tuple[str, int, float]]:
    """Extract LoRA adapter specs from the config."""
    adapters = []
    for name, sub in (("vision", config.vision), ("speech", config.audio)):
        lora_cfg = getattr(sub, "lora", None) if sub is not None else None
        if lora_cfg is None:
            continue
        rank = lora_cfg["r"]
        alpha = lora_cfg["lora_alpha"]
        scale = alpha / rank
        adapters.append((name, rank, scale))
    return adapters


def _make_lora_linear_factory(
    lora_adapters: list[tuple[str, int, float]],
    gate_holder: dict[str, ir.Value] | None = None,
):
    """Create a LoRALinear factory that captures lora_adapters via closure.

    When ``gate_holder`` is provided, every ``LoRALinear`` built by this
    factory shares the same mapping, so the owning model can populate the
    per-modality gate values once (in ``forward``) and have them applied
    across all decoder layers.
    """

    def factory(in_features: int, out_features: int, bias: bool = True) -> LoRALinear:
        return LoRALinear(
            in_features,
            out_features,
            bias=bias,
            lora_adapters=lora_adapters,
            gate_holder=gate_holder,
        )

    return factory


# Phi4MM ``InputMode`` (see HF ``processing_phi4mm.InputMode``):
#   LANGUAGE=0 (no adapter), VISION=1 / VISION_SPEECH=3 (vision adapter),
#   SPEECH=2 (speech adapter).  HF activates exactly one adapter per forward
#   via ``set_lora_adapter`` (modeling_phi4mm.py).  We derive the equivalent
#   per-adapter activation from the special tokens present in ``input_ids``.
def _compute_phi4mm_lora_gates(
    op: OpBuilder,
    input_ids: ir.Value,
    image_token_id: int,
    audio_token_id: int,
    dtype: ir.DataType,
) -> tuple[ir.Value, ir.Value]:
    """Derive ``(vision_gate, speech_gate)`` scalars from ``input_ids``.

    - ``vision`` adapter is active when any image token is present
      (covers ``VISION`` and ``VISION_SPEECH``).
    - ``speech`` adapter is active when any audio token is present *and*
      no image token is present (``SPEECH`` only; ``VISION_SPEECH`` uses
      the vision adapter, matching HF).
    """
    img_tok = op.Constant(value_int=image_token_id)
    aud_tok = op.Constant(value_int=audio_token_id)
    # Reduce over the whole sequence to a 0/1 float scalar.
    has_image = op.ReduceMax(
        op.Cast(op.Equal(input_ids, img_tok), to=ir.DataType.FLOAT), keepdims=False
    )
    has_audio = op.ReduceMax(
        op.Cast(op.Equal(input_ids, aud_tok), to=ir.DataType.FLOAT), keepdims=False
    )
    one = op.Constant(value_float=1.0)
    vision_gate = op.Cast(has_image, to=dtype)
    speech_gate = op.Cast(op.Mul(has_audio, op.Sub(one, has_image)), to=dtype)
    return vision_gate, speech_gate


class _LoRATextModel(TextModel):
    """Text model with LoRA-aware decoder layers."""

    def __init__(
        self, config: ArchitectureConfig, lora_adapters: list[tuple[str, int, float]]
    ):
        nn.Module.__init__(self)
        lora_factory = _make_lora_linear_factory(lora_adapters)
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                DecoderLayer(config, linear_class=lora_factory, mlp_class=FusedGateUpMLP)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)


def _preprocess_phi4mm_weights(
    config: ArchitectureConfig, state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Shared weight preprocessing for Phi4MM models (LoRA + fused weight splitting).

    **LoRA strategy — separate parameters, gated per modality:**
    The ONNX model keeps lora_A.{name}.weight and lora_B.{name}.weight as
    separate initializers.  ``LoRALinear.forward()`` applies them at runtime:
    ``out = base(x) + gate_{name} * scale * x @ A.T @ B.T``.  HuggingFace
    activates exactly one adapter per forward based on the input modality
    (``set_lora_adapter``): vision for image / image+audio, speech for
    audio-only, none for text-only.  In the 4-model split task the embedding
    model derives ``vision_gate``/``speech_gate`` from ``input_ids`` and the
    decoder multiplies each adapter by its gate, reproducing HF exactly.

    Do NOT merge LoRA into the base weight here — keeping adapters separate is
    what allows the per-modality gating above.

    Steps performed:
    1. Strip ``base_layer.`` from LoRA-wrapped weight names so
       ``qkv_proj.base_layer.weight`` → ``qkv_proj.weight``.
    2. Split fused ``qkv_proj`` weights (base + LoRA A/B) into separate
       ``q_proj``, ``k_proj``, ``v_proj`` entries.
    3. Split fused ``gate_up_proj`` weights (base + LoRA A/B) into
       ``gate_proj`` and ``up_proj`` entries.
    4. ``o_proj`` and ``down_proj`` LoRA A/B pass through unchanged
       (not fused, no splitting needed).
    """
    # Strip "base_layer." from LoRA-wrapped weight names.
    # HF stores e.g. "qkv_proj.base_layer.weight"; after stripping this
    # becomes "qkv_proj.weight" and falls through to the split logic below.
    for key in list(state_dict.keys()):
        if ".base_layer." in key:
            new_key = key.replace(".base_layer.", ".")
            state_dict[new_key] = state_dict.pop(key)

    for key in list(state_dict.keys()):
        # Split qkv_proj base weight/bias
        if ("qkv_proj.weight" in key or "qkv_proj.bias" in key) and "lora" not in key:
            w = state_dict.pop(key)
            base = key.split("qkv_proj.")[0]
            suffix = key.split("qkv_proj.")[1]
            q, k, v = split_fused_qkv(
                w,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
            )
            state_dict[f"{base}q_proj.{suffix}"] = q
            state_dict[f"{base}k_proj.{suffix}"] = k
            state_dict[f"{base}v_proj.{suffix}"] = v

        # Split qkv_proj LoRA B weights (output dim split, same layout as base)
        elif "qkv_proj.lora_B." in key:
            w = state_dict.pop(key)
            base = key.split("qkv_proj.")[0]
            suffix = key.split("qkv_proj.")[1]
            q, k, v = split_fused_qkv(
                w,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
            )
            state_dict[f"{base}q_proj.{suffix}"] = q
            state_dict[f"{base}k_proj.{suffix}"] = k
            state_dict[f"{base}v_proj.{suffix}"] = v

        # Split qkv_proj LoRA A weights (same A for q/k/v)
        elif "qkv_proj.lora_A." in key:
            w = state_dict.pop(key)
            base = key.split("qkv_proj.")[0]
            suffix = key.split("qkv_proj.")[1]
            state_dict[f"{base}q_proj.{suffix}"] = w
            state_dict[f"{base}k_proj.{suffix}"] = w.clone()
            state_dict[f"{base}v_proj.{suffix}"] = w.clone()

        # gate_up_proj weights (base and LoRA) pass through unchanged —
        # FusedGateUpMLP keeps the fused weight and splits activations.

    # Weight tying
    if config.tie_word_embeddings:
        embed_key = "model.embed_tokens.weight"
        lm_head_key = "lm_head.weight"
        if lm_head_key not in state_dict and embed_key in state_dict:
            state_dict[lm_head_key] = state_dict[embed_key]

    return state_dict


class Phi4MMCausalLMModel(Phi3CausalLMModel):
    """Phi4-MM text-only model with LoRA adapters.

    LoRA weights are kept as separate parameters in the ONNX model.
    The forward pass computes the LoRA contribution alongside the base linear.

    Replicates HuggingFace's ``Phi4MMForCausalLM``.
    """

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        self.config = config
        lora_adapters = _parse_lora_adapters(config)

        self.model = _LoRATextModel(config, lora_adapters)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return _preprocess_phi4mm_weights(self.config, state_dict)


# -----------------------------------------------------------------------
# Phi4-MM Multimodal Model (Vision + Audio + Text with LoRA)
class _Phi4MMNaViTPatchEmbedding(PatchEmbedding):
    """SigLIP NaViT patch embedding with mask-aware position IDs.

    HF's ``SiglipVisionEmbeddings`` (vision_siglip_navit.py:571) assigns
    position embeddings based on each crop's *valid* patch grid rather than a
    fixed 0..N-1 sequence.  For a crop with ``nb_h`` valid rows and ``nb_w``
    valid cols (the top-left block; padding is on the right/bottom), the valid
    patch at grid (r, c) gets::

        pos_id = floor(r * P / nb_h) * P + floor(c * P / nb_w)

    (``P`` = patches per side).  This is exactly HF's ``bucketize`` of evenly
    spaced fractional coordinates against ``arange(1/P, 1, 1/P)``.  Padded
    patches get position id 0.  Without this redistribution, padded HD
    sub-crops diverge sharply from HF (cos ~0.8) even with attention masking.
    """

    def __init__(self, image_size: int, patch_size: int, hidden_size: int):
        super().__init__(image_size=image_size, patch_size=patch_size, hidden_size=hidden_size)
        self._P = image_size // patch_size  # patches per side

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        patch_attention_mask: ir.Value | None = None,
    ):
        # Conv patchify → (N, P*P, hidden), mirroring PatchEmbedding.
        patches = op.Conv(
            pixel_values,
            self.patch_embedding,
            self.patch_embedding_bias,
            kernel_shape=[self.patch_embedding.shape[2], self.patch_embedding.shape[3]],
            strides=[self.patch_embedding.shape[2], self.patch_embedding.shape[3]],
        )
        batch_size = op.Shape(patches, start=0, end=1)
        hidden_dim = op.Shape(patches, start=1, end=2)
        patches = op.Reshape(
            patches,
            op.Concat(batch_size, hidden_dim, op.Constant(value_ints=[-1]), axis=0),
        )
        patches = op.Transpose(patches, perm=[0, 2, 1])  # (N, P*P, hidden)

        if patch_attention_mask is None:
            return op.Add(patches, self.position_embedding)

        P = self._P  # noqa: N806
        # mask: (N, P, P) {0,1} → counts of valid rows/cols (top-left block).
        mask_i = op.Cast(patch_attention_mask, to=ir.DataType.INT64)
        # nb_h = sum of column 0 over rows; nb_w = sum of row 0 over cols.
        col0 = op.Slice(
            mask_i,
            op.Constant(value_ints=[0]),
            op.Constant(value_ints=[1]),
            op.Constant(value_ints=[2]),
        )  # (N, P, 1)
        nb_h = op.ReduceSum(col0, op.Constant(value_ints=[1]), keepdims=1)  # (N,1,1)
        row0 = op.Slice(
            mask_i,
            op.Constant(value_ints=[0]),
            op.Constant(value_ints=[1]),
            op.Constant(value_ints=[1]),
        )  # (N, 1, P)
        nb_w = op.ReduceSum(row0, op.Constant(value_ints=[2]), keepdims=1)  # (N,1,1)
        # Guard against a fully-padded crop (nb_h or nb_w == 0): clamp the
        # divisor to >=1 to avoid a division by zero.  Such crops have no valid
        # patches, so every position is masked to id 0 by ``valid`` below and
        # the clamped quotient is never used.
        one_i = op.Constant(value_int=1)
        nb_h_div = op.Max(nb_h, one_i)
        nb_w_div = op.Max(nb_w, one_i)

        # r: (1, P, 1), c: (1, 1, P)
        idx = op.Constant(value_ints=list(range(P)))
        r = op.Reshape(idx, op.Constant(value_ints=[1, P, 1]))
        c = op.Reshape(idx, op.Constant(value_ints=[1, 1, P]))
        p_const = op.Constant(value_int=P)  # scalar int64
        # pos_h = floor(r * P / nb_h); pos_w = floor(c * P / nb_w)
        pos_h = op.Div(op.Mul(r, p_const), nb_h_div)  # (N, P, 1)
        pos_w = op.Div(op.Mul(c, p_const), nb_w_div)  # (N, 1, P)
        # pos_id(r,c) = pos_h * P + pos_w → (N, P, P)
        pos_id = op.Add(op.Mul(pos_h, p_const), pos_w)
        # Valid patches are the top-left nb_h x nb_w block; others -> id 0.
        valid = op.And(op.Less(r, nb_h), op.Less(c, nb_w))  # (N, P, P)
        zero = op.Constant(value_int=0)
        pos_id = op.Where(valid, pos_id, zero)
        pos_id_flat = op.Reshape(
            pos_id, op.Concat(batch_size, op.Constant(value_ints=[-1]), axis=0)
        )  # (N, P*P)
        # Gather per-crop position embeddings: (N, P*P, hidden)
        pos_emb = op.Gather(self.position_embedding, pos_id_flat, axis=0)
        return op.Add(patches, pos_emb)


# -----------------------------------------------------------------------


class _Phi4MMSigLIPEncoder(nn.Module):
    """SigLIP vision encoder for Phi4MM (no post_layernorm).

    HuggingFace Phi4MM uses ``layer_idx=-2`` when extracting SigLIP
    features, meaning it uses the *second-to-last* hidden state and
    skips the final encoder layer.  We replicate this by instantiating
    ``num_hidden_layers - 1`` encoder layers.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        image_size = vc.image_size or 336 if vc else 336
        patch_size = vc.patch_size or 14 if vc else 14
        hidden_size = vc.hidden_size or config.hidden_size if vc else config.hidden_size
        intermediate_size = vc.intermediate_size or hidden_size * 4 if vc else hidden_size * 4
        num_heads = vc.num_attention_heads or 4 if vc else 4
        num_layers = vc.num_hidden_layers or 2 if vc else 2
        norm_eps = vc.norm_eps if vc else 1e-6
        self._dtype = config.dtype
        self.embeddings = _Phi4MMNaViTPatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
        )
        # layer_idx=-2: only run the first (num_layers - 1) encoder layers.
        # When num_layers == 1, this gives 0 layers — just patch embeddings,
        # matching HF's layer_idx=-2 behaviour for a 1-layer SigLIP.
        self.encoder = VisionEncoder(
            num_layers=num_layers - 1,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            norm_eps=norm_eps,
        )

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        image_attention_mask: ir.Value | None = None,
    ):
        hidden_states = self.embeddings(
            op, pixel_values, patch_attention_mask=image_attention_mask
        )
        attn_bias = None
        if image_attention_mask is not None:
            # image_attention_mask: (N_crops, H, H) of {0, 1} → flatten to the
            # patch sequence (N_crops, 1, 1, H*H) and build an additive
            # attention bias: 0 for valid patches, large-negative for padding.
            # This replicates HF's ``patch_attention_mask`` so padded patches
            # do not leak into valid patches via self-attention.  ORT's
            # Attention requires the bias' query dim to equal the Q seq length,
            # so broadcast (Expand) the key-padding bias across all queries:
            # (N_crops, 1, 1, S) → (N_crops, 1, S, S).
            n_crops = op.Shape(image_attention_mask, start=0, end=1)  # (1,)
            seq = op.Shape(hidden_states, start=1, end=2)  # (1,) = H*H
            flat_shape = op.Concat(
                n_crops,
                op.Constant(value_ints=[1, 1, -1]),
                axis=0,
            )
            mask_flat = op.Reshape(image_attention_mask, flat_shape)
            mask_flat = op.Cast(mask_flat, to=self._dtype)
            one = op.Cast(op.Constant(value_floats=[1.0]), to=self._dtype)
            neg = op.Cast(op.Constant(value_floats=[-50000.0]), to=self._dtype)
            # (1 - mask) * -50000 → 0 where valid, -50000 where padded.
            attn_bias = op.Mul(op.Sub(one, mask_flat), neg)
            target_shape = op.Concat(n_crops, op.Constant(value_ints=[1]), seq, seq, axis=0)
            attn_bias = op.Expand(attn_bias, target_shape)
        return self.encoder(op, hidden_states, attn_bias)


class _GELUModule(nn.Module):
    """GELU activation as an nn.Module (no parameters)."""

    def forward(self, op: OpBuilder, x: ir.Value):
        return op.Gelu(x)


class _Phi4MMProjectionMLP(nn.Module):
    """Sequential MLP: Linear → GELU → Linear.

    Registers children at string indices "0", "1", "2" to match
    HuggingFace Sequential naming convention. Uses nn.Module with
    indexed setattr rather than subclassing ModuleList to avoid
    doubled path segments in ONNX initializer names.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        layers = [
            Linear(in_features, out_features),
            _GELUModule(),
            Linear(out_features, out_features),
        ]
        for i, layer in enumerate(layers):
            setattr(self, str(i), layer)
        self._layers = layers

    def forward(self, op: OpBuilder, x: ir.Value):
        for layer in self._layers:
            x = layer(op, x)
        return x


class _Phi4MMImageEmbedding(nn.Module):
    """Phi4MM image embedding: SigLIP + projection + HD transform params."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        vision_hidden_size = vc.hidden_size or config.hidden_size if vc else config.hidden_size
        text_hidden_size = config.hidden_size

        self.img_processor = _Phi4MMSigLIPEncoder(config)
        self.img_projection = _Phi4MMProjectionMLP(vision_hidden_size, text_hidden_size)
        self.glb_GN = nn.Parameter([1, 1, vision_hidden_size])
        self.sub_GN = nn.Parameter([1, 1, 1, vision_hidden_size])

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        vision_features = self.img_processor(op, pixel_values)
        return self.img_projection(op, vision_features)


class _Phi4MMAudioEmbedding(nn.Module):
    """Phi4MM audio embedding: Conformer encoder + speech/vision projections."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        audio = config.audio
        audio_dim = (audio.attention_dim if audio else None) or 256
        text_hidden_size = config.hidden_size

        self.encoder = ConformerEncoder(
            input_size=(audio.input_size if audio else None) or 80,
            attention_dim=audio_dim,
            attention_heads=(audio.attention_heads if audio else None) or 4,
            num_blocks=(audio.num_blocks if audio else None) or 2,
            linear_units=(audio.linear_units if audio else None) or 1024,
            kernel_size=(audio.kernel_size if audio else None) or 3,
            conv_channels=(audio.conv_channels if audio else None) or audio_dim,
            t5_bias_max_distance=((audio.t5_bias_max_distance if audio else None) or 500),
        )

        # Audio projection "speech" and "vision" branches.
        # Named directly to match ONNX initializer paths; the HF
        # "audio_projection." prefix is stripped in preprocess_weights.
        self.speech = _Phi4MMProjectionMLP(audio_dim, text_hidden_size)
        self.vision = _Phi4MMProjectionMLP(audio_dim, text_hidden_size)

    def forward(self, op: OpBuilder, audio_features: ir.Value):
        audio_hidden = self.encoder(op, audio_features)
        return self.speech(op, audio_hidden)


class _Phi4MMImageAudioEmbedding(nn.Module):
    """Combined image + audio embedding (embed_tokens_extend)."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.image_embed = _Phi4MMImageEmbedding(config)
        self.audio_embed = _Phi4MMAudioEmbedding(config)

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value | None = None,
        audio_features: ir.Value | None = None,
    ):
        image_embeddings = None
        if pixel_values is not None:
            image_embeddings = self.image_embed(op, pixel_values)
        audio_embeddings = None
        if audio_features is not None:
            audio_embeddings = self.audio_embed(op, audio_features)
        return image_embeddings, audio_embeddings


class _Phi4MMMultiModalTextModel(nn.Module):
    """Phi4MM inner model: text embeddings + multimodal mixing + LoRA decoder."""

    def __init__(
        self,
        config: ArchitectureConfig,
        lora_adapters: list[tuple[str, int, float]],
    ):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        lora_factory = _make_lora_linear_factory(lora_adapters)
        self.embed_tokens_extend = _Phi4MMImageAudioEmbedding(config)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(config, linear_class=lora_factory, mlp_class=FusedGateUpMLP)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)
        self._image_mixer = InputMixer(image_token_id=config.image_token_id or 200010)
        self._audio_mixer = InputMixer(
            image_token_id=(config.audio.token_id if config.audio else None) or 200011
        )

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        pixel_values: ir.Value | None = None,
        audio_features: ir.Value | None = None,
    ):
        has_multimodal = pixel_values is not None or audio_features is not None
        if has_multimodal:
            text_embeddings = self.embed_tokens(op, input_ids)
            image_embeddings, audio_embeddings = self.embed_tokens_extend(
                op, pixel_values=pixel_values, audio_features=audio_features
            )
            hidden_states = text_embeddings
            if image_embeddings is not None:
                hidden_states = self._image_mixer(
                    op, hidden_states, image_embeddings, input_ids
                )
            if audio_embeddings is not None:
                hidden_states = self._audio_mixer(
                    op, hidden_states, audio_embeddings, input_ids
                )
        else:
            hidden_states = self.embed_tokens(op, input_ids)

        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


# -----------------------------------------------------------------------
# Phi4-MM Four-Model Split Sub-Modules
# -----------------------------------------------------------------------


class _Phi4MMVisionModel(nn.Module):
    """Phi4MM vision encoder: SigLIP + AvgPool + HD transform + projection MLP.

    Takes raw pixel values (all crops) and image_sizes, encodes each crop
    through SigLIP, applies 2x spatial compression (AvgPool2d), then assembles
    the HD transform: arranges sub-crops in a spatial grid, adds row separator
    tokens (sub_GN), and concatenates with the global crop feature in
    sub-first order (sub | glb_GN | global).  The combined sequence is
    projected to the text decoder's hidden dimension.

    HF reference: ``Phi4MMImageEmbedding.forward`` with
    ``image_token_compression_cls="avg_pool_2d"`` and
    ``hd_transform_order="sub_glb"``.

    Inputs:
        pixel_values: (N_crops, 3, image_size, image_size)
        image_sizes: (1, 2) — [height_px, width_px] of the original image,
            used to derive the sub-crop grid dimensions
            (h = height_px // crop_size, w = width_px // crop_size).
    Outputs:
        image_features: (total_image_tokens, text_hidden_size)
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        vision_hidden = (vc.hidden_size if vc else None) or config.hidden_size
        text_hidden = config.hidden_size
        image_size = (vc.image_size if vc else None) or 448
        patch_size = (vc.patch_size if vc else None) or 14

        # H: patches per side from SigLIP (e.g. 448/14 = 32)
        # Hp: patches per side after AvgPool2d(kernel=2, stride=2) = H // 2 = 16
        # crop_size: pixel width/height of one crop tile (= image_size for Phi4MM)
        self._H = image_size // patch_size  # 32
        self._Hp = self._H // 2  # 16 — post-AvgPool spatial dimension
        self._C = vision_hidden  # 1152
        self._crop_size = image_size  # 448

        self.img_processor = _Phi4MMSigLIPEncoder(config)
        self.img_projection = _Phi4MMProjectionMLP(vision_hidden, text_hidden)
        self.glb_GN = nn.Parameter([1, 1, vision_hidden])
        self.sub_GN = nn.Parameter([1, 1, 1, vision_hidden])

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        image_sizes: ir.Value,
        image_attention_mask: ir.Value | None = None,
    ):
        """Encode crops through SigLIP + HD spatial reassembly.

        Steps:
          1. SigLIP: (N_crops, 3, H_px, W_px) → (N_crops, H*H, C)
          2. AvgPool2d(kernel=2, stride=2): 32x32 -> 16x16 per crop
          3. Extract grid ratio (h, w) from image_sizes
          4. Global crop (index 0): reshape to (1, Hp, Hp, C) and append
             sub_GN row separators → (1, Hp*(Hp+1), C)
          5. Sub crops (indices 1..h*w): arrange in (h, w) grid →
             (1, h*Hp, w*Hp, C), optionally crop padded rows/cols using
             ``image_attention_mask``, then append sub_GN row separators
          6. Assemble sub-first: [sub | glb_GN | global]
          7. Project flat sequence through img_projection MLP

        When ``image_attention_mask`` (N_crops, H, H) is provided, padded
        sub-crops are cropped to their useful height/width before the row
        separators are added — replicating HuggingFace's masked HD transform
        (``modeling_phi4mm.py`` lines 376-387).  Without it, the full
        (h*Hp, w*Hp) grid is emitted, which only matches HF when no sub-crop
        is padded.  Phi4MM's processor always emits the mask, so for parity
        (and correct multi-image token alignment) it must be passed.
        """
        H = self._H  # noqa: N806  # 32 patches per side from SigLIP
        Hp = self._Hp  # noqa: N806  # 16 patches per side after AvgPool
        C = self._C  # noqa: N806  # 1152 vision hidden dim

        # ── Step 1: SigLIP encode all crops ────────────────────────────
        # pixel_values: (N_crops, 3, 448, 448)
        # vision_features: (N_crops, H*H, C) = (N_crops, 1024, 1152)
        # The patch mask excludes padded patches from self-attention so the
        # valid patches' features match HF (which passes patch_attention_mask).
        vision_features = self.img_processor(
            op, pixel_values, image_attention_mask=image_attention_mask
        )

        # ── Step 2: AvgPool2d — compress 32x32 patches to 16x16 ───────
        # Reshape: (N_crops, H*H, C) -> (N_crops, H, H, C)
        n_crops = op.Shape(vision_features, start=0, end=1)  # (1,) int64
        feats = op.Reshape(
            vision_features,
            op.Concat(n_crops, op.Constant(value_ints=[H, H, C]), axis=0),
        )  # (N_crops, 32, 32, 1152)
        # NHWC -> NCHW for AveragePool
        feats = op.Transpose(feats, perm=[0, 3, 1, 2])  # (N_crops, 1152, 32, 32)
        feats = op.AveragePool(feats, kernel_shape=[2, 2], strides=[2, 2])
        # (N_crops, 1152, 16, 16)
        feats = op.Transpose(feats, perm=[0, 2, 3, 1])  # (N_crops, 16, 16, 1152)
        # Flatten spatial dims: (N_crops, Hp*Hp, C)
        feats = op.Reshape(
            feats,
            op.Concat(n_crops, op.Constant(value_ints=[Hp * Hp, C]), axis=0),
        )  # (N_crops, 256, 1152)

        # ── Step 3: Derive h, w grid dimensions from image_sizes ──────
        # image_sizes: (1, 2) = [[height_px, width_px]]
        # h = height_px // crop_size, w = width_px // crop_size
        img_hw = op.Reshape(image_sizes, op.Constant(value_ints=[-1]))  # (2,)
        crop_size_t = op.Constant(value_int=self._crop_size)  # scalar int64
        h = op.Div(
            op.Gather(img_hw, op.Constant(value_int=0), axis=0), crop_size_t
        )  # scalar — num sub-crop rows
        w = op.Div(
            op.Gather(img_hw, op.Constant(value_int=1), axis=0), crop_size_t
        )  # scalar — num sub-crop cols
        B_ = op.Mul(h, w)  # noqa: N806  # total number of sub crops

        # ── Step 4: Split global crop (index 0) and sub crops ─────────
        # global: (1, Hp*Hp, C) = (1, 256, 1152)
        global_feat = op.Slice(
            feats,
            op.Constant(value_ints=[0]),
            op.Constant(value_ints=[1]),
            op.Constant(value_ints=[0]),
        )

        # sub: (h*w, Hp*Hp, C) — slices crops 1 through h*w (inclusive)
        B_plus_1 = op.Unsqueeze(op.Add(B_, op.Constant(value_int=1)), [0])  # noqa: N806
        sub_feat = op.Slice(
            feats,
            op.Constant(value_ints=[1]),
            B_plus_1,  # end = h*w + 1
            op.Constant(value_ints=[0]),
        )  # (h*w, 256, 1152)

        # ── Step 5: Process global crop — reshape + row separators ────
        # Reshape to 2D grid: (1, Hp, Hp, C)
        global_4d = op.Reshape(global_feat, op.Constant(value_ints=[1, Hp, Hp, C]))
        # sub_GN: (1, 1, 1, C) -> tile to (1, Hp, 1, C) as row separators
        temp_glb_gn = op.Tile(
            self.sub_GN, op.Constant(value_ints=[1, Hp, 1, 1])
        )  # (1, 16, 1, 1152)
        # Append one separator per row: (1, Hp, Hp+1, C)
        glb_rows = op.Concat(global_4d, temp_glb_gn, axis=2)
        # Flatten rows: (1, Hp*(Hp+1), C) = (1, 272, 1152)
        glb_img = op.Reshape(glb_rows, op.Constant(value_ints=[1, -1, C]))

        # ── Step 6: Process sub crops — grid layout + row separators ──
        # Reshape each crop to 2D grid: (h*w, Hp, Hp, C)
        sub_4d_shape = op.Concat(
            op.Unsqueeze(B_, [0]),
            op.Constant(value_ints=[Hp, Hp, C]),
            axis=0,
        )
        sub_4d = op.Reshape(sub_feat, sub_4d_shape)  # (h*w, 16, 16, 1152)

        # Arrange in (h, w) spatial grid: (1, h, w, Hp, Hp, C)
        sub_6d_shape = op.Concat(
            op.Constant(value_ints=[1]),
            op.Unsqueeze(h, [0]),
            op.Unsqueeze(w, [0]),
            op.Constant(value_ints=[Hp, Hp, C]),
            axis=0,
        )
        sub_6d = op.Reshape(sub_4d, sub_6d_shape)  # (1, h, w, 16, 16, 1152)

        # Permute row/col blocks to place spatially adjacent:
        # (1, h, w, Hp, Hp, C) -> (1, h, Hp, w, Hp, C)
        sub_6d_t = op.Transpose(sub_6d, perm=[0, 1, 3, 2, 4, 5])

        # Flatten block dims: (1, h*Hp, w*Hp, C)
        h_hp = op.Mul(h, op.Constant(value_int=Hp))  # h * 16
        w_hp = op.Mul(w, op.Constant(value_int=Hp))  # w * 16
        sub_flat_shape = op.Concat(
            op.Constant(value_ints=[1]),
            op.Unsqueeze(h_hp, [0]),
            op.Unsqueeze(w_hp, [0]),
            op.Constant(value_ints=[C]),
            axis=0,
        )
        sub_grid = op.Reshape(sub_6d_t, sub_flat_shape)  # (1, h*16, w*16, 1152)

        # ── Optional mask crop: drop padded sub-crop rows/cols ────────
        # HF (modeling_phi4mm.py:376-382) uses image_attention_mask to crop
        # the assembled grid to its useful height/width before adding the
        # row separators.  The mask is at the pre-AvgPool resolution (HxH);
        # HF samples it stride-2 (``[..., 0::2, 0::2]``) so each value maps
        # to one post-AvgPool position (HpxHp per crop).
        if image_attention_mask is not None:
            # Sub-crop masks: crops 1..B_ → (h*w, H, H)
            sub_mask = op.Slice(
                image_attention_mask,
                op.Constant(value_ints=[1]),
                B_plus_1,
                op.Constant(value_ints=[0]),
            )  # (h*w, 32, 32)
            # Stride-2 downsample to match AvgPool: (h*w, Hp, Hp)
            sub_mask = op.Slice(
                sub_mask,
                op.Constant(value_ints=[0, 0]),
                op.Constant(value_ints=[H, H]),
                op.Constant(value_ints=[1, 2]),
                op.Constant(value_ints=[2, 2]),
            )  # (h*w, 16, 16)
            # Arrange in spatial grid mirroring the feature layout:
            # (1, h, w, Hp, Hp) -> (1, h, Hp, w, Hp) -> (1, h*Hp, w*Hp)
            mask_5d_shape = op.Concat(
                op.Constant(value_ints=[1]),
                op.Unsqueeze(h, [0]),
                op.Unsqueeze(w, [0]),
                op.Constant(value_ints=[Hp, Hp]),
                axis=0,
            )
            mask_6d = op.Reshape(sub_mask, mask_5d_shape)
            mask_6d_t = op.Transpose(mask_6d, perm=[0, 1, 3, 2, 4])
            mask_grid = op.Reshape(
                mask_6d_t,
                op.Concat(
                    op.Constant(value_ints=[1]),
                    op.Unsqueeze(h_hp, [0]),
                    op.Unsqueeze(w_hp, [0]),
                    axis=0,
                ),
            )  # (1, h*Hp, w*Hp)
            # useful_height = sum of column 0; useful_width = sum of row 0
            col0 = op.Slice(
                mask_grid,
                op.Constant(value_ints=[0]),
                op.Constant(value_ints=[1]),
                op.Constant(value_ints=[2]),
            )  # (1, h*Hp, 1)
            row0 = op.Slice(
                mask_grid,
                op.Constant(value_ints=[0]),
                op.Constant(value_ints=[1]),
                op.Constant(value_ints=[1]),
            )  # (1, 1, w*Hp)
            useful_h = op.Cast(
                op.ReduceSum(col0, op.Constant(value_ints=[0, 1, 2]), keepdims=0),
                to=ir.DataType.INT64,
            )  # scalar
            useful_w = op.Cast(
                op.ReduceSum(row0, op.Constant(value_ints=[0, 1, 2]), keepdims=0),
                to=ir.DataType.INT64,
            )  # scalar
            uh = op.Unsqueeze(useful_h, [0])  # (1,)
            uw = op.Unsqueeze(useful_w, [0])  # (1,)
            # Crop grid: (1, useful_h, useful_w, C)
            sub_grid = op.Slice(
                sub_grid,
                op.Constant(value_ints=[0, 0]),
                op.Concat(uh, uw, axis=0),
                op.Constant(value_ints=[1, 2]),
            )
            # Row separators tiled to useful_height instead of h*Hp.
            sub_sep_tile = op.Concat(
                op.Constant(value_ints=[1]),
                uh,
                op.Constant(value_ints=[1, 1]),
                axis=0,
            )
            temp_sub_gn = op.Tile(self.sub_GN, sub_sep_tile)  # (1, useful_h, 1, C)
            sub_rows = op.Concat(sub_grid, temp_sub_gn, axis=2)
            sub_img = op.Reshape(sub_rows, op.Constant(value_ints=[1, -1, C]))
        else:
            # Row separators: sub_GN (1, 1, 1, C) tiled to (1, h*Hp, 1, C)
            sub_sep_tile = op.Concat(
                op.Constant(value_ints=[1]),
                op.Unsqueeze(h_hp, [0]),
                op.Constant(value_ints=[1, 1]),
                axis=0,
            )
            temp_sub_gn = op.Tile(self.sub_GN, sub_sep_tile)  # (1, h*16, 1, 1152)
            # Append one separator per row: (1, h*Hp, w*Hp+1, C)
            sub_rows = op.Concat(sub_grid, temp_sub_gn, axis=2)
            # Flatten rows: (1, h*Hp*(w*Hp+1), C)
            sub_img = op.Reshape(sub_rows, op.Constant(value_ints=[1, -1, C]))

        # ── Step 7: Assemble sub-first and project ────────────────────
        # HF hd_transform_order = "sub_glb": [sub | glb_GN | global]
        # glb_GN: (1, 1, C) — separator between sub and global features
        full_seq = op.Concat(sub_img, self.glb_GN, glb_img, axis=1)
        # (1, total_tokens, C) -> (total_tokens, C)
        flat = op.Reshape(full_seq, op.Constant(value_ints=[-1, C]))
        # Project to text hidden dim: (total_tokens, text_hidden)
        return self.img_projection(op, flat)


class _Phi4MMSpeechModel(nn.Module):
    """Phi4MM speech encoder: Conformer + projection MLPs.

    Encodes mel spectrogram audio features through a Conformer encoder
    and projects to the text decoder's hidden dimension. Includes both
    "speech" and "vision" projection branches, selected at runtime by
    ``audio_projection_mode``.

    Inputs:
        audio_embeds: [batch, audio_seq_len, num_mel_bins]
        audio_sizes: [num_audio_clips] — number of frames per clip
        audio_projection_mode: scalar int — 0=speech branch, 1=vision branch
    Outputs:
        audio_features: [num_speech_tokens, hidden_size]
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        audio = config.audio
        audio_dim = (audio.attention_dim if audio else None) or 256
        text_hidden = config.hidden_size

        self.encoder = ConformerEncoder(
            input_size=(audio.input_size if audio else None) or 80,
            attention_dim=audio_dim,
            attention_heads=(audio.attention_heads if audio else None) or 4,
            num_blocks=(audio.num_blocks if audio else None) or 2,
            linear_units=(audio.linear_units if audio else None) or 1024,
            kernel_size=(audio.kernel_size if audio else None) or 3,
            conv_channels=(audio.conv_channels if audio else None) or audio_dim,
            t5_bias_max_distance=(audio.t5_bias_max_distance if audio else None) or 500,
        )

        # Both projection branches for speech-only and combined modes.
        # Named "speech"/"vision" directly; the HF "audio_projection."
        # prefix is stripped in preprocess_weights since onnxscript
        # doesn't propagate intermediate nn.Module container names.
        self.speech = _Phi4MMProjectionMLP(audio_dim, text_hidden)
        self.vision = _Phi4MMProjectionMLP(audio_dim, text_hidden)

    def forward(
        self,
        op: OpBuilder,
        audio_embeds: ir.Value,
        audio_sizes: ir.Value,
        audio_projection_mode: ir.Value,
    ):
        # audio_sizes is plumbed for the I/O contract with ORT GenAI.
        # It will be used for variable-length batching in a follow-up.
        audio_hidden = self.encoder(op, audio_embeds)

        # Both branches run (ONNX graphs are static), then select via mode
        speech_branch = self.speech(op, audio_hidden)
        vision_branch = self.vision(op, audio_hidden)

        # audio_projection_mode: 0=speech, 1=vision
        is_vision_mode = op.Equal(audio_projection_mode, 1)
        return op.Where(is_vision_mode, vision_branch, speech_branch)


class _Phi4MMEmbeddingModel(nn.Module):
    """Phi4MM embedding: token embedding + InputMixer fusion.

    Embeds text tokens and replaces image/audio placeholder positions
    with the corresponding projected features from the vision and
    speech encoders.

    Inputs:
        input_ids: [batch, seq_len]
        image_features: [num_image_tokens, hidden_size]
        audio_features: [num_speech_tokens, hidden_size]
    Outputs:
        inputs_embeds: [batch, seq_len, hidden_size]
        vision_gate: scalar — 1.0 if any image token present else 0.0
        speech_gate: scalar — 1.0 if audio present and no image else 0.0
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self._image_token_id = config.image_token_id or 200010
        self._audio_token_id = (config.audio.token_id if config.audio else None) or 200011
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
        )
        self._image_mixer = InputMixer(image_token_id=config.image_token_id or 200010)
        self._audio_mixer = InputMixer(
            image_token_id=(config.audio.token_id if config.audio else None) or 200011
        )

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        image_features: ir.Value,
        audio_features: ir.Value,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
        # Add batch dim: [num_tokens, hidden] → [1, num_tokens, hidden]
        # InputMixer expects [batch, seq, hidden] for GatherElements
        image_features_3d = op.Unsqueeze(image_features, [0])
        audio_features_3d = op.Unsqueeze(audio_features, [0])
        hidden_states = self._image_mixer(op, hidden_states, image_features_3d, input_ids)
        hidden_states = self._audio_mixer(op, hidden_states, audio_features_3d, input_ids)
        # Derive per-modality LoRA gates here (the decoder only sees
        # ``inputs_embeds`` and cannot recover the modality from input_ids).
        vision_gate, speech_gate = _compute_phi4mm_lora_gates(
            op, input_ids, self._image_token_id, self._audio_token_id, self._dtype
        )
        return hidden_states, vision_gate, speech_gate


class _Phi4MMDecoderModel(nn.Module):
    """Phi4MM text decoder with LoRA adapters.

    Takes fused input embeddings (text + vision + audio) and runs
    through the transformer decoder with LoRA-adapted attention and
    MLP layers.

    Inputs:
        inputs_embeds: [batch, seq_len, hidden_size]
        attention_mask: [batch, past_seq_len + seq_len]
        position_ids: [batch, seq_len]
        past_key_values: list of (key, value) tuples
        vision_gate: scalar LoRA gate for the vision adapter (optional)
        speech_gate: scalar LoRA gate for the speech adapter (optional)
    Outputs:
        logits: [batch, seq_len, vocab_size]
        present_key_values: list of (key, value) tuples
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        lora_adapters = _parse_lora_adapters(config)
        # Shared mapping {adapter_name: gate ir.Value}, populated in forward()
        # so every LoRALinear gates its adapter by the active input modality.
        self._lora_gates: dict[str, ir.Value] = {}
        lora_factory = _make_lora_linear_factory(lora_adapters, self._lora_gates)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(config, linear_class=lora_factory, mlp_class=FusedGateUpMLP)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        vision_gate: ir.Value | None = None,
        speech_gate: ir.Value | None = None,
    ):
        # Publish the per-modality LoRA gates (computed in the embedding model
        # from input_ids) so every LoRALinear in the decoder applies only the
        # adapter active for the current modality.  Clear any gates from a
        # previous forward so a stale gate cannot leak across calls.
        self._lora_gates.clear()
        if vision_gate is not None:
            self._lora_gates["vision"] = vision_gate
        if speech_gate is not None:
            self._lora_gates["speech"] = speech_gate

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(op, position_ids)
        # Use inputs_embeds for query_length since decoder has no input_ids
        attention_bias = create_attention_bias(
            op,
            input_ids=inputs_embeds,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values


class Phi4MMMultiModalModel(nn.Module):
    """Phi-4 Multimodal model (4-model split).

    Produces four separate ONNX models via ``Phi4MMMultiModalTask``:

    - ``vision``: SigLIP encoder + projection -> image_features
    - ``speech``: Conformer encoder + projection -> audio_features
    - ``embedding``: token embedding + InputMixer fusion -> inputs_embeds
    - ``decoder``: LoRA text decoder + lm_head -> logits + KV cache

    Replicates HuggingFace's ``Phi4MMForCausalLM``.
    """

    default_task: str = "phi4mm-multimodal"
    category: str = "Multimodal"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.vision_encoder = _Phi4MMVisionModel(config)
        self.speech_encoder = _Phi4MMSpeechModel(config)
        self.embedding = _Phi4MMEmbeddingModel(config)
        self.decoder = _Phi4MMDecoderModel(config)

    def forward(self, op: OpBuilder, **kwargs):
        raise NotImplementedError(
            "Phi4MMMultiModalModel uses Phi4MMMultiModalTask "
            "which calls each sub-module (vision_encoder, "
            "speech_encoder, embedding, decoder) separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Preprocess and remap weights for the 4-model split.

        First applies LoRA unwrapping and fused weight splitting,
        then remaps HuggingFace prefixes to 4-model structure:

        - ``model.embed_tokens_extend.image_embed.*``
          -> ``vision_encoder.*``
        - ``model.embed_tokens_extend.audio_embed.*``
          -> ``speech_encoder.*``
        - ``model.embed_tokens.*``
          -> ``embedding.embed_tokens.*``
        - ``model.layers.*`` -> ``decoder.layers.*``
        - ``model.norm.*`` -> ``decoder.norm.*``
        - ``lm_head.*`` -> ``decoder.lm_head.*``

        Layer-count filtering: weights for layer indices beyond the truncated
        layer counts (``config.num_hidden_layers``, ``config.vision.num_hidden_layers``,
        ``config.audio.num_blocks``) are dropped here rather than showing as
        UNEXPECTED warnings during ``apply_weights``.  This is expected when the
        ONNX model is built with fewer layers than the full checkpoint (e.g. 2
        text layers vs. the default 32).
        """
        state_dict = _preprocess_phi4mm_weights(self.config, state_dict)

        # Fix vision position embedding: 3D [1,N,H] -> 2D [N,H]
        for key in list(state_dict.keys()):
            if "img_processor.embeddings.position_embedding.weight" in key:
                if state_dict[key].dim() == 3:
                    state_dict[key] = state_dict[key].squeeze(0)

        # Remap prefixes to 4-model structure
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = _remap_phi4mm_weight_key(key)
            renamed[new_key] = value

        # Drop weights for truncated layers so they don't appear as UNEXPECTED
        # in apply_weights when the ONNX model has fewer layers than the checkpoint.
        renamed = _drop_truncated_layer_weights(renamed, self.config)

        # Duplicate embed_tokens weight for decoder (tied weights)
        embed_key = "embedding.embed_tokens.weight"
        lm_head_key = "decoder.lm_head.weight"
        if self.config.tie_word_embeddings:
            if embed_key in renamed and lm_head_key not in renamed:
                renamed[lm_head_key] = renamed[embed_key]

        return renamed


def _layer_index_from_key(key: str, prefix: str) -> int | None:
    """Return the layer index N from a key of the form ``{prefix}.{N}.rest``.

    Returns ``None`` if the key doesn't match or N is not an integer.
    """
    if not key.startswith(prefix + "."):
        return None
    rest = key[len(prefix) + 1 :]
    idx_str, _, _ = rest.partition(".")
    try:
        return int(idx_str)
    except ValueError:
        return None


def _drop_truncated_layer_weights(
    renamed: dict[str, torch.Tensor],
    config: ArchitectureConfig,
) -> dict[str, torch.Tensor]:
    """Drop weights for layers/blocks beyond the truncated layer counts.

    When the ONNX model is built with fewer layers than the full checkpoint
    (e.g. ``num_text_layers=2`` against a 32-layer checkpoint) the extra
    layer weights would otherwise all appear as UNEXPECTED in ``apply_weights``.
    Dropping them here silences those spurious warnings.

    Affected namespaces after prefix remapping:
    - ``decoder.layers.{N}.*``       — drop if N >= config.num_hidden_layers
    - ``vision_encoder.img_processor.encoder.layers.{N}.*``
                                      — drop if N >= config.vision.num_hidden_layers
    - ``speech_encoder.encoder.encoders.{N}.*``
                                      — drop if N >= config.audio.num_blocks
    """
    max_decoder = config.num_hidden_layers
    # _Phi4MMSigLIPEncoder builds (num_hidden_layers - 1) layers (layer_idx=-2),
    # so drop weights for layers at or above that count.
    max_vision = (
        config.vision.num_hidden_layers - 1
        if config.vision is not None and config.vision.num_hidden_layers is not None
        else None
    )
    max_audio = (
        config.audio.num_blocks
        if config.audio is not None and config.audio.num_blocks is not None
        else None
    )

    filtered: dict[str, torch.Tensor] = {}
    for key, value in renamed.items():
        # Decoder layers
        idx = _layer_index_from_key(key, "decoder.layers")
        if idx is not None and idx >= max_decoder:
            continue

        # Vision encoder layers
        if max_vision is not None:
            idx = _layer_index_from_key(key, "vision_encoder.img_processor.encoder.layers")
            if idx is not None and idx >= max_vision:
                continue

        # Audio encoder blocks
        if max_audio is not None:
            idx = _layer_index_from_key(key, "speech_encoder.encoder.encoders")
            if idx is not None and idx >= max_audio:
                continue

        filtered[key] = value
    return filtered


def _remap_phi4mm_weight_key(key: str) -> str:
    """Remap a single HuggingFace weight key to 4-model prefix."""
    # Vision encoder: image_embed sub-tree
    img_prefix = "model.embed_tokens_extend.image_embed."
    if key.startswith(img_prefix):
        suffix = key[len(img_prefix) :]
        # SigLIP vision encoder uses fc1/fc2 for MLP — rename to up_proj/down_proj
        suffix = suffix.replace(".mlp.fc1.", ".mlp.up_proj.")
        suffix = suffix.replace(".mlp.fc2.", ".mlp.down_proj.")
        return "vision_encoder." + suffix

    # Speech encoder: audio_embed sub-tree
    audio_prefix = "model.embed_tokens_extend.audio_embed."
    if key.startswith(audio_prefix):
        suffix = key[len(audio_prefix) :]
        # Strip "audio_projection." since onnxscript resolves the speech
        # and vision projection branches directly on the module (Bug 5).
        suffix = suffix.removeprefix("audio_projection.")
        # Strip PyTorch activation-checkpoint wrapper segment injected by
        # gradient-checkpointing — not present in the ONNX module tree.
        suffix = suffix.replace("._checkpoint_wrapped_module", "")
        return "speech_encoder." + suffix

    # Token embeddings -> embedding model
    if key == "model.embed_tokens.weight":
        return "embedding.embed_tokens.weight"

    # Decoder layers
    if key.startswith("model.layers."):
        return "decoder." + key[len("model.") :]

    # Final norm
    if key.startswith("model.norm."):
        return "decoder." + key[len("model.") :]

    # LM head
    if key.startswith("lm_head."):
        return "decoder." + key

    return key


class Phi3SmallCausalLMModel(Phi3CausalLMModel):
    """Phi3-Small model with block-sparse attention.

    Uses block-sparse attention with alternating dense/sparse layers,
    MuP (maximal update parameterization) scaling, and GeGELU activation
    with clamping.

    Replicates HuggingFace's ``Phi3SmallForCausalLM``.
    """

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        state_dict = super().preprocess_weights(state_dict)
        q_size = self.config.num_attention_heads * self.config.head_dim
        kv_size = self.config.num_key_value_heads * self.config.head_dim
        num_kv_groups = self.config.num_attention_heads // self.config.num_key_value_heads
        group_size = num_kv_groups + 2

        for key in list(state_dict.keys()):
            # Handle combined query_key_value projection
            if "query_key_value.weight" in key:
                weight = state_dict.pop(key)
                wqkv = weight.t().reshape(
                    self.config.hidden_size,
                    self.config.num_key_value_heads,
                    group_size,
                    self.config.head_dim,
                )
                q_weight = (
                    wqkv[:, :, :num_kv_groups, :].reshape(self.config.hidden_size, q_size).t()
                )
                k_weight = wqkv[:, :, [-2], :].reshape(self.config.hidden_size, kv_size).t()
                v_weight = wqkv[:, :, [-1], :].reshape(self.config.hidden_size, kv_size).t()

                prefix = key.replace("query_key_value.weight", "")
                state_dict[f"{prefix}q_proj.weight"] = q_weight
                state_dict[f"{prefix}k_proj.weight"] = k_weight
                state_dict[f"{prefix}v_proj.weight"] = v_weight

            elif "query_key_value.bias" in key:
                bias = state_dict.pop(key)
                bias_grouped = bias.reshape(
                    self.config.num_key_value_heads,
                    group_size,
                    self.config.head_dim,
                )
                q_bias = bias_grouped[:, :num_kv_groups, :].reshape(q_size)
                k_bias = bias_grouped[:, [-2], :].reshape(kv_size)
                v_bias = bias_grouped[:, [-1], :].reshape(kv_size)

                prefix = key.replace("query_key_value.bias", "")
                state_dict[f"{prefix}q_proj.bias"] = q_bias
                state_dict[f"{prefix}k_proj.bias"] = k_bias
                state_dict[f"{prefix}v_proj.bias"] = v_bias

        return state_dict
