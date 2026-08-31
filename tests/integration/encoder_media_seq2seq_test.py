# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Integration tests for encoder, media, and sequence-to-sequence model parity."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import transformers

from integration._support import (
    _make_session,
)
from mobius import build, models
from mobius._configs import ArchitectureConfig
from mobius._testing.comparison import (
    assert_logits_close,
)

_ENCODER_MODELS = [
    pytest.param("google-bert/bert-base-uncased", False, id="bert-base"),
    pytest.param("distilbert/distilbert-base-uncased", False, id="distilbert-base"),
    pytest.param("facebook/esm2_t6_8M_UR50D", False, id="esm2-8m"),
    pytest.param(
        "FacebookAI/roberta-base",
        False,
        id="roberta-base",
        marks=pytest.mark.skip(
            reason="RoBERTa position_ids differ from BERT; see test_roberta_hidden_states_parity"
        ),
    ),
    pytest.param(
        "albert/albert-base-v2",
        False,
        id="albert-base",
        marks=pytest.mark.skip(
            reason="ALBERT embedding_size != hidden_size not yet supported (LayerNorm shape mismatch)"
        ),
    ),
]


_UNEQUAL_LENGTH_PROMPTS = {
    "protein": [
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    ],
    "text": [
        "The capital of France is Paris.",
        (
            "Encoder models read the whole sequence at once, so padding must not "
            "change a row's contextual embeddings."
        ),
    ],
}


@pytest.mark.integration
@pytest.mark.integration_fast
@pytest.mark.parametrize("model_id,trust_remote_code", _ENCODER_MODELS)
class TestEncoderOnlyForward:
    """Compare encoder-only hidden states between ONNX and PyTorch."""

    def test_hidden_states_match(self, model_id: str, trust_remote_code: bool):
        """Forward pass: input_ids → last_hidden_state."""
        from mobius._testing.torch_reference import (
            load_torch_encoder_model,
            torch_encoder_forward,
        )

        onnx_model = build(model_id, dtype="f32", load_weights=True)
        torch_model, tokenizer = load_torch_encoder_model(model_id)

        prompt = "The capital of France is Paris."
        tokens = tokenizer(prompt, return_tensors="np")
        input_ids = tokens["input_ids"].astype(np.int64)
        attention_mask = tokens["attention_mask"].astype(np.int64)
        token_type_ids = tokens.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.astype(np.int64)

        torch_hidden = torch_encoder_forward(
            torch_model, input_ids, attention_mask, token_type_ids
        )

        session = _make_session(onnx_model)
        feeds: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            feeds["token_type_ids"] = token_type_ids
        elif "token_type_ids" in session.input_names:
            # Model requires token_type_ids but tokenizer doesn't provide
            # it (e.g. RoBERTa). Use zeros.
            feeds["token_type_ids"] = np.zeros_like(input_ids)
        onnx_outputs = session.run(feeds)
        session.close()

        assert_logits_close(
            onnx_outputs["last_hidden_state"],
            torch_hidden,
            rtol=1e-3,
            atol=1e-3,
        )

    def test_padded_batch_matches_huggingface_and_unpadded_rows(
        self, model_id: str, trust_remote_code: bool
    ):
        """Batch two unequal-length inputs and check every row three ways.

        This is the assertion the attention-mask path lives or dies on. An
        encoder that hands the raw rank-2 mask to ``op.Attention`` cannot even
        run at batch > 1 -- the mask fails to broadcast to
        ``(batch, heads, q, kv)`` -- and were it to run it would add a 0/1 bias
        where a large negative one is required, so padded positions would leak
        into every valid residue. Checking each row against both HuggingFace
        and its own unpadded run separates "the export is wrong" from "the
        export is right but padding-sensitive".
        """
        from mobius._testing.torch_reference import (
            load_torch_encoder_model,
            torch_encoder_forward,
        )

        onnx_model = build(model_id, dtype="f32", load_weights=True)
        torch_model, tokenizer = load_torch_encoder_model(model_id)
        prompts = _UNEQUAL_LENGTH_PROMPTS["protein" if "esm" in model_id.lower() else "text"]
        session = _make_session(onnx_model)

        def feeds_for(texts: list[str]) -> dict[str, np.ndarray]:
            batch = tokenizer(texts, return_tensors="np", padding=True)
            feeds: dict[str, np.ndarray] = {
                "input_ids": batch["input_ids"].astype(np.int64),
                "attention_mask": batch["attention_mask"].astype(np.int64),
            }
            if "token_type_ids" in session.input_names:
                token_type_ids = batch.get("token_type_ids")
                feeds["token_type_ids"] = (
                    np.zeros_like(feeds["input_ids"])
                    if token_type_ids is None
                    else token_type_ids.astype(np.int64)
                )
            return feeds

        try:
            batched = feeds_for(prompts)
            mask = batched["attention_mask"]
            assert mask.shape[0] == 2
            assert mask.min() == 0, "prompts must differ in length so the batch pads"

            batched_out = session.run(batched)["last_hidden_state"]
            torch_hidden = torch_encoder_forward(
                torch_model,
                batched["input_ids"],
                mask,
                batched.get("token_type_ids"),
            )

            for row, prompt in enumerate(prompts):
                length = int(mask[row].sum())
                got, ref = batched_out[row, :length], torch_hidden[row, :length]

                # Scale-free metrics rather than an elementwise tolerance. A
                # contextual embedding is used as a direction: what matters is
                # that the vector matches, not that every one of its small
                # components survives fp32 cancellation through six layers.
                rel_l2 = np.linalg.norm(got - ref) / np.linalg.norm(ref)
                cosine = (got * ref).sum(-1) / (
                    np.linalg.norm(got, axis=-1) * np.linalg.norm(ref, axis=-1)
                )
                assert rel_l2 < 1e-3, f"row {row}: relative L2 error {rel_l2:.2e}"
                assert cosine.min() > 1 - 1e-5, (
                    f"row {row}: worst per-token cosine {cosine.min():.8f}"
                )

                solo = session.run(feeds_for([prompt]))["last_hidden_state"]
                # Padding is bit-exact, not merely close: a padded position that
                # reached a valid one would show up here as a real difference.
                np.testing.assert_array_equal(solo[0, :length], got)
        finally:
            session.close()


_SEQ2SEQ_MODELS = [
    pytest.param("facebook/bart-base", False, id="bart-base"),
    pytest.param("google-t5/t5-small", False, id="t5-small"),
    pytest.param(
        "Helsinki-NLP/opus-mt-en-de",
        False,
        id="marian-en-de",
        marks=pytest.mark.skip(reason="HF repo has no safetensors (pytorch_model.bin only)"),
    ),
]


@pytest.mark.integration
@pytest.mark.integration_fast
@pytest.mark.parametrize("model_id,trust_remote_code", _SEQ2SEQ_MODELS)
class TestSeq2SeqForward:
    """Compare seq2seq encoder/decoder between ONNX and PyTorch."""

    def test_encoder_hidden_states_match(self, model_id: str, trust_remote_code: bool):
        """Encoder forward: input_ids → last_hidden_state."""
        from mobius import build
        from mobius._testing.torch_reference import (
            load_torch_seq2seq_model,
            torch_seq2seq_encoder_forward,
        )

        pkg = build(model_id, dtype="f32", load_weights=True)
        torch_model, tokenizer = load_torch_seq2seq_model(model_id)

        prompt = "Translate English to French: The house is wonderful."
        tokens = tokenizer(prompt, return_tensors="np")
        input_ids = tokens["input_ids"].astype(np.int64)
        attention_mask = tokens["attention_mask"].astype(np.int64)

        torch_enc = torch_seq2seq_encoder_forward(torch_model, input_ids, attention_mask)

        encoder_session = _make_session(pkg["encoder"])
        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        onnx_enc = encoder_session.run(feeds)
        encoder_session.close()

        assert_logits_close(
            onnx_enc["last_hidden_state"],
            torch_enc,
            rtol=1e-3,
            atol=1e-3,
        )

    def test_decoder_prefill_logits_match(self, model_id: str, trust_remote_code: bool):
        """Decoder prefill: decoder_input_ids + encoder_hidden_states → logits."""
        from mobius import build
        from mobius._testing.torch_reference import (
            load_torch_seq2seq_model,
            torch_seq2seq_decoder_forward,
            torch_seq2seq_encoder_forward,
        )

        pkg = build(model_id, dtype="f32", load_weights=True)
        torch_model, tokenizer = load_torch_seq2seq_model(model_id)

        # Encode
        src = "Translate English to French: The house is wonderful."
        src_tokens = tokenizer(src, return_tensors="np")
        input_ids = src_tokens["input_ids"].astype(np.int64)
        attention_mask = src_tokens["attention_mask"].astype(np.int64)
        enc_hidden = torch_seq2seq_encoder_forward(torch_model, input_ids, attention_mask)

        # Decoder input (start token)
        decoder_start_id = torch_model.config.decoder_start_token_id
        if decoder_start_id is None:
            decoder_start_id = tokenizer.pad_token_id
        decoder_input_ids = np.array([[decoder_start_id]], dtype=np.int64)

        torch_logits, _ = torch_seq2seq_decoder_forward(
            torch_model, decoder_input_ids, enc_hidden, attention_mask
        )

        # ONNX decoder
        hf_config = transformers.AutoConfig.from_pretrained(model_id)
        num_decoder_layers = getattr(
            hf_config, "num_decoder_layers", hf_config.num_hidden_layers
        )
        num_heads = hf_config.num_attention_heads
        head_dim = hf_config.d_model // num_heads

        decoder_session = _make_session(pkg["decoder"])
        feeds: dict[str, np.ndarray] = {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": enc_hidden,
            "attention_mask": attention_mask,
        }
        for i in range(num_decoder_layers):
            feeds[f"past_key_values.{i}.self.key"] = np.zeros(
                (1, num_heads, 0, head_dim), dtype=np.float32
            )
            feeds[f"past_key_values.{i}.self.value"] = np.zeros(
                (1, num_heads, 0, head_dim), dtype=np.float32
            )
            feeds[f"past_key_values.{i}.cross.key"] = np.zeros(
                (1, num_heads, 0, head_dim), dtype=np.float32
            )
            feeds[f"past_key_values.{i}.cross.value"] = np.zeros(
                (1, num_heads, 0, head_dim), dtype=np.float32
            )
        onnx_out = decoder_session.run(feeds)
        decoder_session.close()

        assert_logits_close(onnx_out["logits"], torch_logits, rtol=1e-3, atol=1e-3)


_VISION_MODELS = [
    pytest.param("google/vit-base-patch16-224", False, id="vit-base"),
    pytest.param(
        "facebook/dinov2-small",
        False,
        id="dinov2-small",
        marks=pytest.mark.skip(
            reason="DINOv2 uses layer_scale (lambda) not yet implemented in ViT model"
        ),
    ),
    pytest.param(
        "microsoft/beit-base-patch16-224",
        False,
        id="beit-base",
        marks=pytest.mark.skip(
            reason="BeiT uses layer scale and relative position bias not implemented in ViT model; k_proj has no bias in HF"
        ),
    ),
]


@pytest.mark.integration
@pytest.mark.integration_fast
@pytest.mark.parametrize("model_id,trust_remote_code", _VISION_MODELS)
class TestVisionForward:
    """Compare vision model hidden states between ONNX and PyTorch."""

    def test_hidden_states_match(self, model_id: str, trust_remote_code: bool):
        """Forward pass: pixel_values → last_hidden_state."""
        from mobius._testing.torch_reference import (
            load_torch_vision_model,
            torch_vision_forward,
        )

        onnx_model = build(model_id, dtype="f32", load_weights=True)
        torch_model, _processor = load_torch_vision_model(model_id)

        # Random image input — use model config image_size as the
        # authoritative source (processor.size may differ, e.g. DINOv2)
        rng = np.random.default_rng(42)
        hf_config = transformers.AutoConfig.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        image_size = getattr(hf_config, "image_size", 224)
        pixel_values = rng.standard_normal((1, 3, image_size, image_size)).astype(np.float32)

        torch_hidden = torch_vision_forward(torch_model, pixel_values)

        session = _make_session(onnx_model)
        feeds = {"pixel_values": pixel_values}
        onnx_outputs = session.run(feeds)
        session.close()

        assert_logits_close(
            onnx_outputs["last_hidden_state"],
            torch_hidden,
            rtol=1e-3,
            atol=1e-3,
        )


_AUDIO_MODELS = [
    pytest.param(
        "facebook/wav2vec2-base",
        False,
        id="wav2vec2-base",
        marks=pytest.mark.skip(
            reason="Model files no longer available on HF Hub (no safetensors)"
        ),
    ),
    pytest.param(
        "facebook/hubert-base-ls960",
        False,
        id="hubert-base",
        marks=pytest.mark.skip(
            reason="Model files no longer available on HF Hub (no safetensors)"
        ),
    ),
]


@pytest.mark.integration
@pytest.mark.integration_fast
@pytest.mark.parametrize("model_id,trust_remote_code", _AUDIO_MODELS)
class TestAudioForward:
    """Compare audio model hidden states between ONNX and PyTorch."""

    def test_hidden_states_match(self, model_id: str, trust_remote_code: bool):
        """Forward pass: input_values → last_hidden_state."""
        from mobius._testing.torch_reference import (
            load_torch_audio_model,
            torch_audio_forward,
        )

        onnx_model = build(model_id, dtype="f32", load_weights=True)
        torch_model, _processor = load_torch_audio_model(model_id)

        # Random audio waveform (1 second at 16kHz)
        rng = np.random.default_rng(42)
        input_values = rng.standard_normal((1, 16000)).astype(np.float32)

        torch_hidden = torch_audio_forward(torch_model, input_values)

        session = _make_session(onnx_model)
        feeds = {"input_values": input_values}
        onnx_outputs = session.run(feeds)
        session.close()

        assert_logits_close(
            onnx_outputs["last_hidden_state"],
            torch_hidden,
            rtol=1e-3,
            atol=1e-3,
        )


_WHISPER_MODELS = [
    pytest.param("openai/whisper-tiny", False, id="whisper-tiny"),
]


@pytest.mark.integration
@pytest.mark.integration_fast
@pytest.mark.parametrize("model_id,trust_remote_code", _WHISPER_MODELS)
class TestWhisperForward:
    """Compare Whisper encoder + decoder between ONNX and PyTorch."""

    def test_encoder_hidden_states_match(self, model_id: str, trust_remote_code: bool):
        """Encoder forward: input_features → encoder_hidden_states."""
        from mobius._testing.torch_reference import (
            load_torch_whisper_model,
            torch_whisper_encoder_forward,
        )

        pkg = build(model_id, dtype="f32", load_weights=True)
        torch_model, processor = load_torch_whisper_model(model_id)

        # Random mel spectrogram input (1 second of audio → 80 mel bins)
        rng = np.random.default_rng(42)
        num_mel_bins = processor.feature_extractor.feature_size
        # Whisper expects 30s of audio → 3000 frames after feature extraction
        audio_seq_len = 3000
        input_features = rng.standard_normal((1, num_mel_bins, audio_seq_len)).astype(
            np.float32
        )

        torch_hidden = torch_whisper_encoder_forward(torch_model, input_features)

        encoder_session = _make_session(pkg["encoder"])
        feeds = {"input_features": input_features}
        onnx_enc = encoder_session.run(feeds)
        encoder_session.close()

        assert_logits_close(
            onnx_enc["encoder_hidden_states"],
            torch_hidden,
            rtol=1e-3,
            atol=1e-3,
        )

    def test_decoder_prefill_logits_match(self, model_id: str, trust_remote_code: bool):
        """Decoder prefill: decoder_input_ids + encoder_hidden_states → logits."""
        from mobius._testing.torch_reference import (
            load_torch_whisper_model,
            torch_whisper_decoder_forward,
            torch_whisper_encoder_forward,
        )

        pkg = build(model_id, dtype="f32", load_weights=True)
        torch_model, processor = load_torch_whisper_model(model_id)

        # Get encoder hidden states from a random mel spectrogram
        rng = np.random.default_rng(42)
        num_mel_bins = processor.feature_extractor.feature_size
        audio_seq_len = 3000
        input_features = rng.standard_normal((1, num_mel_bins, audio_seq_len)).astype(
            np.float32
        )
        enc_hidden = torch_whisper_encoder_forward(torch_model, input_features)

        # Decoder input (start-of-transcript token)
        decoder_start_id = torch_model.config.decoder_start_token_id
        if decoder_start_id is None:
            decoder_start_id = 50258  # Whisper default SOT token
        decoder_input_ids = np.array([[decoder_start_id]], dtype=np.int64)

        torch_logits, _ = torch_whisper_decoder_forward(
            torch_model, decoder_input_ids, enc_hidden
        )

        # ONNX decoder forward
        hf_config = transformers.AutoConfig.from_pretrained(model_id)
        num_decoder_layers = hf_config.decoder_layers
        num_heads = hf_config.decoder_attention_heads
        head_dim = hf_config.d_model // num_heads
        position_ids = np.zeros((1, 1), dtype=np.int64)

        decoder_session = _make_session(pkg["decoder"])
        feeds: dict[str, np.ndarray] = {
            "decoder_input_ids": decoder_input_ids,
            "encoder_hidden_states": enc_hidden,
            "position_ids": position_ids,
        }
        for i in range(num_decoder_layers):
            feeds[f"past_key_values.{i}.key"] = np.zeros(
                (1, num_heads, 0, head_dim), dtype=np.float32
            )
            feeds[f"past_key_values.{i}.value"] = np.zeros(
                (1, num_heads, 0, head_dim), dtype=np.float32
            )
        onnx_out = decoder_session.run(feeds)
        decoder_session.close()

        assert_logits_close(onnx_out["logits"], torch_logits, rtol=1e-3, atol=1e-3)


def _make_encoder_feeds(
    seq_len: int = 8,
    vocab_size: int = 256,
    type_vocab_size: int = 2,
) -> dict[str, np.ndarray]:
    """Create input feeds for encoder-only models."""
    rng = np.random.default_rng(42)
    input_ids = rng.integers(1, vocab_size, size=(1, seq_len)).astype(np.int64)
    attention_mask = np.ones((1, seq_len), dtype=np.int64)
    feeds: dict[str, np.ndarray] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if type_vocab_size > 0:
        feeds["token_type_ids"] = np.zeros((1, seq_len), dtype=np.int64)
    return feeds


@pytest.mark.integration
@pytest.mark.integration_fast
def test_bert_hidden_states_parity():
    """BERT encoder: random-weight hidden states match HuggingFace."""
    import onnx_ir as ir
    from transformers import BertConfig
    from transformers import BertModel as HFBertModel

    from mobius import build_from_module
    from mobius.integrations._weight_loading import apply_weights

    # Tiny BERT config
    hf_cfg = BertConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=256,
        max_position_embeddings=128,
        type_vocab_size=2,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    )
    ref_model = HFBertModel._from_config(hf_cfg).float().eval()

    # Matching ONNX config
    config = ArchitectureConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        vocab_size=256,
        max_position_embeddings=128,
        type_vocab_size=2,
        hidden_act="gelu",
        rms_norm_eps=1e-12,
        pad_token_id=0,
    )
    config.dtype = ir.DataType.FLOAT

    onnx_module = models.BertModel(config)
    pkg = build_from_module(onnx_module, config, task="feature-extraction")
    onnx_model = pkg["model"]

    # Transfer weights: HF state_dict -> preprocess -> apply
    preprocessed = onnx_module.preprocess_weights(dict(ref_model.state_dict()))
    apply_weights(onnx_model, preprocessed)

    # Run both models
    feeds = _make_encoder_feeds(seq_len=8, vocab_size=256, type_vocab_size=2)
    input_ids = feeds["input_ids"]
    attention_mask = feeds["attention_mask"]
    token_type_ids = feeds["token_type_ids"]

    with torch.no_grad():
        hf_out = ref_model(
            input_ids=torch.from_numpy(input_ids),
            attention_mask=torch.from_numpy(attention_mask),
            token_type_ids=torch.from_numpy(token_type_ids),
        )
        hf_hidden = hf_out.last_hidden_state.numpy()

    session = _make_session(onnx_model)
    onnx_out = session.run(feeds)
    session.close()

    assert_logits_close(onnx_out["last_hidden_state"], hf_hidden, rtol=1e-3, atol=1e-3)


@pytest.mark.integration
@pytest.mark.integration_fast
def test_distilbert_hidden_states_parity():
    """DistilBERT encoder: random-weight hidden states match HuggingFace."""
    import onnx_ir as ir
    from transformers import (
        DistilBertConfig,
    )
    from transformers import (
        DistilBertModel as HFDistilBertModel,
    )

    from mobius import build_from_module
    from mobius.integrations._weight_loading import apply_weights

    # Tiny DistilBERT config
    hf_cfg = DistilBertConfig(
        dim=64,
        hidden_dim=128,
        n_heads=4,
        n_layers=2,
        vocab_size=256,
        max_position_embeddings=128,
        activation="gelu",
        qa_dropout=0.0,
        seq_classif_dropout=0.0,
    )
    ref_model = HFDistilBertModel._from_config(hf_cfg).float().eval()

    # Matching ONNX config
    config = ArchitectureConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        vocab_size=256,
        max_position_embeddings=128,
        type_vocab_size=0,
        hidden_act="gelu",
        rms_norm_eps=1e-5,
        pad_token_id=0,
    )
    config.dtype = ir.DataType.FLOAT

    onnx_module = models.DistilBertModel(config)
    pkg = build_from_module(onnx_module, config, task="feature-extraction")
    onnx_model = pkg["model"]

    # Transfer weights
    preprocessed = onnx_module.preprocess_weights(dict(ref_model.state_dict()))
    apply_weights(onnx_model, preprocessed)

    # DistilBERT doesn't use token_type_ids but the ONNX graph
    # declares it as input, so we must provide zeros
    feeds = _make_encoder_feeds(seq_len=8, vocab_size=256, type_vocab_size=1)

    input_ids = feeds["input_ids"]
    attention_mask = feeds["attention_mask"]

    with torch.no_grad():
        hf_out = ref_model(
            input_ids=torch.from_numpy(input_ids),
            attention_mask=torch.from_numpy(attention_mask),
        )
        hf_hidden = hf_out.last_hidden_state.numpy()

    session = _make_session(onnx_model)
    onnx_out = session.run(feeds)
    session.close()

    assert_logits_close(onnx_out["last_hidden_state"], hf_hidden, rtol=1e-3, atol=1e-3)


@pytest.mark.integration
@pytest.mark.integration_fast
def test_roberta_hidden_states_parity():
    """RoBERTa encoder: random-weight hidden states match HuggingFace."""
    import onnx_ir as ir
    from transformers import RobertaConfig
    from transformers import RobertaModel as HFRobertaModel

    from mobius import build_from_module
    from mobius.integrations._weight_loading import apply_weights

    # Tiny RoBERTa config: type_vocab_size=1, pad_token_id=1
    hf_cfg = RobertaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=256,
        max_position_embeddings=130,
        type_vocab_size=1,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        pad_token_id=1,
    )
    ref_model = HFRobertaModel._from_config(hf_cfg).float().eval()

    # Matching ONNX config -- BertModel handles both BERT and RoBERTa
    config = ArchitectureConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        vocab_size=256,
        max_position_embeddings=130,
        type_vocab_size=1,
        hidden_act="gelu",
        rms_norm_eps=1e-5,
        pad_token_id=1,
    )
    config.dtype = ir.DataType.FLOAT

    # RoBERTa uses the same BertModel class
    onnx_module = models.BertModel(config)
    pkg = build_from_module(onnx_module, config, task="feature-extraction")
    onnx_model = pkg["model"]

    # Transfer weights -- preprocess_weights strips "roberta." prefix
    preprocessed = onnx_module.preprocess_weights(dict(ref_model.state_dict()))
    apply_weights(onnx_model, preprocessed)

    # RoBERTa uses type_vocab_size=1 (all zeros)
    feeds = _make_encoder_feeds(seq_len=8, vocab_size=256, type_vocab_size=1)

    input_ids = feeds["input_ids"]
    attention_mask = feeds["attention_mask"]
    token_type_ids = feeds["token_type_ids"]

    # HF RoBERTa computes position_ids with pad_token_id offset.
    # Our ONNX model always uses 0-based positions. Pass explicit
    # position_ids to HF so both use the same 0-indexed positions.
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len).unsqueeze(0)

    with torch.no_grad():
        hf_out = ref_model(
            input_ids=torch.from_numpy(input_ids),
            attention_mask=torch.from_numpy(attention_mask),
            token_type_ids=torch.from_numpy(token_type_ids),
            position_ids=position_ids,
        )
        hf_hidden = hf_out.last_hidden_state.numpy()

    session = _make_session(onnx_model)
    onnx_out = session.run(feeds)
    session.close()

    assert_logits_close(onnx_out["last_hidden_state"], hf_hidden, rtol=1e-3, atol=1e-3)
