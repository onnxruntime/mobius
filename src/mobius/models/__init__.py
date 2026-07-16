# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

__all__ = [
    "ApertusCausalLMModel",
    "ArceeCausalLMModel",
    "AutoencoderKLModel",
    "AutoencoderKLQwenImageModel",
    "BambaCausalLMModel",
    "BartForConditionalGeneration",
    "BertModel",
    "Blip2Model",
    "BloomCausalLMModel",
    "CLIPVisionModel",
    "SigLIPVisionModel",
    "CTRLCausalLMModel",
    "CausalLMModel",
    "FusedGateUpCausalLMModel",
    "ChatGLMCausalLMModel",
    "CodeGenCausalLMModel",
    "CogVideoXTransformer3DModel",
    "CohereCausalLMModel",
    "ControlNetModel",
    "DeepSeekOCR2CausalLMModel",
    "DeepSeekV3CausalLMModel",
    "DFlashDraftModel",
    "Eagle3DraftModel",
    "DiTTransformer2DModel",
    "DiffLlamaCausalLMModel",
    "DistilBertModel",
    "DogeCausalLMModel",
    "EncDecRNNTModel",
    "Ernie45MoECausalLMModel",
    "ErnieCausalLMModel",
    "ExaOne4CausalLMModel",
    "FalconCausalLMModel",
    "FluxTransformer2DModel",
    "FunASRForConditionalGeneration",
    "GPT2CausalLMModel",
    "GPTJCausalLMModel",
    "GPTNeoXCausalLMModel",
    "GPTNeoXJapaneseCausalLMModel",
    "GPTOSSCausalLMModel",
    "Gemma2CausalLMModel",
    "Gemma3CausalLMModel",
    "Gemma3MultiModalModel",
    "Gemma3nCausalLMModel",
    "Gemma4AssistantCausalLMModel",
    "Gemma4CausalLMModel",
    "Gemma4Model",
    "Gemma4UnifiedModel",
    "GemmaCausalLMModel",
    "Glm4CausalLMModel",
    "Glm4MoECausalLMModel",
    "GlmCausalLMModel",
    "GlmMoeDsaCausalLMModel",
    "GraniteCausalLMModel",
    "GraniteMoECausalLMModel",
    "GraniteMoeHybridCausalLMModel",
    "HunYuanMoEV1CausalLMModel",
    "HunYuanV1DenseCausalLMModel",
    "HunYuanVLMoTModel",
    "HunyuanDiT2DModel",
    "IPAdapterModel",
    "InternLM2CausalLMModel",
    "InternVL2Model",
    "JambaCausalLMModel",
    "JetMoeCausalLMModel",
    "Llama4CausalLMModel",
    "LLaDAModel",
    "LLaVAModel",
    "LayerNormCausalLMModel",
    "LongcatFlashCausalLMModel",
    "MPTCausalLMModel",
    "Mamba2CausalLMModel",
    "MambaCausalLMModel",
    "MiniMaxCausalLMModel",
    "MimiModel",
    "MoshiDepformerModel",
    "MoshiTemporalModel",
    "MoECausalLMModel",
    "NanoChatCausalLMModel",
    "NemotronCausalLMModel",
    "NemotronHCausalLMModel",
    "OLMo2CausalLMModel",
    "OLMoCausalLMModel",
    "OPTCausalLMModel",
    "PersimmonCausalLMModel",
    "Phi3CausalLMModel",
    "Phi3MoECausalLMModel",
    "Phi3SmallCausalLMModel",
    "Phi4MMCausalLMModel",
    "Phi4MMMultiModalModel",
    "PhiCausalLMModel",
    "Qwen25VLCausalLMModel",
    "Qwen25VLDecoderModel",
    "Qwen25VLEmbeddingModel",
    "Qwen25VLTextModel",
    "Qwen25VLVisionEncoderModel",
    "Qwen2MoECausalLMModel",
    "Qwen2VLCausalLMModel",
    "Qwen2VLVisionEncoderModel",
    "Qwen35CausalLMModel",
    "Qwen35MoECausalLMModel",
    "Qwen35MtpModel",
    "Qwen35MoEVL3ModelCausalLMModel",
    "Qwen35VL3ModelCausalLMModel",
    "Qwen35VLDecoderModel",
    "Qwen35VLTextModel",
    "Qwen3ASRForConditionalGeneration",
    "Qwen3CausalLMModel",
    "Qwen3NextCausalLMModel",
    "SenseVoiceSmallModel",
    "Qwen3TTSCodePredictorModel",
    "Qwen3TTSCodecDecoderModel",
    "Qwen3TTSCodecEncoderModel",
    "Qwen3TTSEmbeddingModel",
    "Qwen3TTSForConditionalGeneration",
    "Qwen3TTSSpeakerEncoderModel",
    "Qwen3TTSTalkerDecoderModel",
    "Qwen3TTSTokenizerV2Model",
    "Qwen3VL3ModelCausalLMModel",
    "Qwen3VLCausalLMModel",
    "Qwen3VLDecoderModel",
    "Qwen3VLEmbeddingModel",
    "Qwen3VLTextModel",
    "Qwen3VLVisionEncoderModel",
    "QwenCausalLMModel",
    "QwenImageTransformer2DModel",
    "SD3Transformer2DModel",
    "SmolLM3CausalLMModel",
    "StarCoder2CausalLMModel",
    "T2IAdapterModel",
    "T5ForConditionalGeneration",
    "UNet2DConditionModel",
    "load_unet_lora_safetensors",
    "remap_diffusers_unet_lora",
    "ViTModel",
    "VideoAutoencoderModel",
    "Wav2Vec2ForCTCModel",
    "Wav2Vec2Model",
    "WhisperForConditionalGeneration",
    "XLMCausalLMModel",
    "Zamba2CausalLMModel",
    "mimi_default_config",
    "moshi_depformer_config",
    "moshi_temporal_config",
]

from mobius.models.adapters import IPAdapterModel, T2IAdapterModel
from mobius.models.apertus import ApertusCausalLMModel
from mobius.models.arcee import ArceeCausalLMModel
from mobius.models.bamba import BambaCausalLMModel
from mobius.models.bart import BartForConditionalGeneration
from mobius.models.base import CausalLMModel, FusedGateUpCausalLMModel, LayerNormCausalLMModel
from mobius.models.bert import BertModel
from mobius.models.blip2 import Blip2Model
from mobius.models.chatglm import ChatGLMCausalLMModel
from mobius.models.clip import CLIPVisionModel, SigLIPVisionModel
from mobius.models.cogvideox import CogVideoXTransformer3DModel
from mobius.models.cohere import CohereCausalLMModel
from mobius.models.controlnet import ControlNetModel
from mobius.models.ctrl import CTRLCausalLMModel
from mobius.models.deepseek import DeepSeekV3CausalLMModel
from mobius.models.deepseek_ocr2 import DeepSeekOCR2CausalLMModel
from mobius.models.dflash import DFlashDraftModel
from mobius.models.diffllama import DiffLlamaCausalLMModel
from mobius.models.distilbert import DistilBertModel
from mobius.models.dit import DiTTransformer2DModel
from mobius.models.doge import DogeCausalLMModel
from mobius.models.eagle3 import Eagle3DraftModel
from mobius.models.ernie import ErnieCausalLMModel
from mobius.models.exaone4 import ExaOne4CausalLMModel
from mobius.models.falcon import BloomCausalLMModel, FalconCausalLMModel, MPTCausalLMModel
from mobius.models.flux_sd3 import FluxTransformer2DModel, SD3Transformer2DModel
from mobius.models.fun_asr import FunASRForConditionalGeneration
from mobius.models.gemma import Gemma2CausalLMModel, GemmaCausalLMModel
from mobius.models.gemma3 import Gemma3MultiModalModel
from mobius.models.gemma3_text import Gemma3CausalLMModel
from mobius.models.gemma3n import Gemma3nCausalLMModel
from mobius.models.gemma4 import (
    Gemma4CausalLMModel,
    Gemma4Model,
    Gemma4UnifiedModel,
)
from mobius.models.gemma4_assistant import Gemma4AssistantCausalLMModel
from mobius.models.glm import Glm4CausalLMModel, GlmCausalLMModel
from mobius.models.glm_moe_dsa import GlmMoeDsaCausalLMModel
from mobius.models.gpt2 import GPT2CausalLMModel
from mobius.models.gpt_neox import GPTNeoXCausalLMModel, GPTNeoXJapaneseCausalLMModel
from mobius.models.gptj_codegen import CodeGenCausalLMModel, GPTJCausalLMModel
from mobius.models.gptoss import GPTOSSCausalLMModel
from mobius.models.granite import GraniteCausalLMModel, GraniteMoECausalLMModel
from mobius.models.granitemoehybrid import GraniteMoeHybridCausalLMModel
from mobius.models.hunyuan_dit import HunyuanDiT2DModel
from mobius.models.hunyuan_v1 import HunYuanV1DenseCausalLMModel
from mobius.models.hunyuan_vl_mot import HunYuanVLMoTModel
from mobius.models.internlm import InternLM2CausalLMModel
from mobius.models.internvl import InternVL2Model
from mobius.models.jamba import JambaCausalLMModel
from mobius.models.jetmoe import JetMoeCausalLMModel
from mobius.models.llada import LLaDAModel
from mobius.models.llama4 import Llama4CausalLMModel
from mobius.models.llava import LLaVAModel
from mobius.models.longcat_flash import LongcatFlashCausalLMModel
from mobius.models.mamba import Mamba2CausalLMModel, MambaCausalLMModel
from mobius.models.mimi import MimiModel, mimi_default_config
from mobius.models.minimax import MiniMaxCausalLMModel
from mobius.models.moe import (
    Ernie45MoECausalLMModel,
    Glm4MoECausalLMModel,
    HunYuanMoEV1CausalLMModel,
    MoECausalLMModel,
    Phi3MoECausalLMModel,
    Qwen2MoECausalLMModel,
)
from mobius.models.moshi import (
    MoshiDepformerModel,
    MoshiTemporalModel,
    moshi_depformer_config,
    moshi_temporal_config,
)
from mobius.models.nanochat import NanoChatCausalLMModel
from mobius.models.nemo_rnnt import EncDecRNNTModel
from mobius.models.nemotron import NemotronCausalLMModel
from mobius.models.nemotron_h import NemotronHCausalLMModel
from mobius.models.olmo import OLMo2CausalLMModel, OLMoCausalLMModel
from mobius.models.opt import OPTCausalLMModel
from mobius.models.persimmon import PersimmonCausalLMModel
from mobius.models.phi import (
    Phi3SmallCausalLMModel,
    Phi4MMCausalLMModel,
    Phi4MMMultiModalModel,
    PhiCausalLMModel,
)
from mobius.models.phi3 import Phi3CausalLMModel
from mobius.models.qwen import (
    Qwen3CausalLMModel,
    QwenCausalLMModel,
)
from mobius.models.qwen3_asr import (
    Qwen3ASRForConditionalGeneration,
)
from mobius.models.qwen3_next import Qwen3NextCausalLMModel
from mobius.models.qwen3_tts import (
    Qwen3TTSCodePredictorModel,
    Qwen3TTSEmbeddingModel,
    Qwen3TTSForConditionalGeneration,
    Qwen3TTSSpeakerEncoderModel,
    Qwen3TTSTalkerDecoderModel,
)
from mobius.models.qwen3_tts_tokenizer import (
    Qwen3TTSCodecDecoderModel,
    Qwen3TTSCodecEncoderModel,
    Qwen3TTSTokenizerV2Model,
)
from mobius.models.qwen35 import (
    Qwen35CausalLMModel,
    Qwen35MoECausalLMModel,
    Qwen35MoEVL3ModelCausalLMModel,
    Qwen35VL3ModelCausalLMModel,
    Qwen35VLDecoderModel,
    Qwen35VLTextModel,
)
from mobius.models.qwen35_mtp import Qwen35MtpModel
from mobius.models.qwen_image import QwenImageTransformer2DModel
from mobius.models.qwen_image_vae import AutoencoderKLQwenImageModel
from mobius.models.qwen_vl import (
    Qwen2VLCausalLMModel,
    Qwen2VLVisionEncoderModel,
    Qwen3VL3ModelCausalLMModel,
    Qwen3VLCausalLMModel,
    Qwen3VLDecoderModel,
    Qwen3VLEmbeddingModel,
    Qwen3VLTextModel,
    Qwen3VLVisionEncoderModel,
    Qwen25VLCausalLMModel,
    Qwen25VLDecoderModel,
    Qwen25VLEmbeddingModel,
    Qwen25VLTextModel,
    Qwen25VLVisionEncoderModel,
)
from mobius.models.sensevoice_small import SenseVoiceSmallModel
from mobius.models.smollm import SmolLM3CausalLMModel
from mobius.models.starcoder2 import StarCoder2CausalLMModel
from mobius.models.t5 import T5ForConditionalGeneration
from mobius.models.unet import (
    UNet2DConditionModel,
    load_unet_lora_safetensors,
    remap_diffusers_unet_lora,
)
from mobius.models.vae import AutoencoderKLModel
from mobius.models.video_vae import VideoAutoencoderModel
from mobius.models.vit import ViTModel
from mobius.models.wav2vec2 import Wav2Vec2Model
from mobius.models.wav2vec2_ctc import Wav2Vec2ForCTCModel
from mobius.models.whisper import WhisperForConditionalGeneration
from mobius.models.xlm import XLMCausalLMModel
from mobius.models.zamba2 import Zamba2CausalLMModel
