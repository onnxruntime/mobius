# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Model tasks that define graph I/O structure for different use cases.

A ModelTask encapsulates how to wire an onnxscript.nn.Module into an ONNX
graph: what inputs to create, how to invoke the module, and how to name
the outputs. Different tasks produce models with different I/O contracts.

Example::

    from mobius.tasks import CausalLMTask
    from mobius import build_from_module

    task = CausalLMTask()
    model = build_from_module(my_module, config, task=task)
"""

from __future__ import annotations

__all__ = [
    "AdapterTask",
    "AudioCTCTask",
    "AudioFeatureExtractionTask",
    "CausalLMTask",
    "SmallThinkerGGUFCausalLMTask",
    "CTCAsrTask",
    "FeatureCTCAsrTask",
    "RNNTTask",
    "CodecTask",
    "ComponentSpec",
    "ControlNetTask",
    "DeepSeekV4Task",
    "DFlashDraftTask",
    "Eagle3DraftTask",
    "Qwen35MtpTask",
    "Qwen4ExpCausalLMTask",
    "Qwen4ExpVisionLanguageTask",
    "DenoisingTask",
    "DiarizationTask",
    "FeatureExtractionTask",
    "GGUFEncoderFeatureExtractionTask",
    "GGUFEmbeddingFeatureExtractionTask",
    "GGUFProjectorVisionLanguageTask",
    "GGUFSpeakerProjectorModel",
    "GGUFSpeakerProjectorTask",
    "GGUFVisionProjectorModel",
    "GGUFVisionProjectorTask",
    "FunASRSpeechLanguageTask",
    "Gemma3VisionLanguageTask",
    "Gemma3nTask",
    "Gemma4AssistantTask",
    "Gemma4Task",
    "Gemma4UnifiedTask",
    "Gemma4TextCausalLMTask",
    "GlmMoeDsaTask",
    "GlmOcrVLTask",
    "HybridCausalLMTask",
    "HyV3MtpTask",
    "FalconH1CausalLMTask",
    "Cosmos3EdgeVLTask",
    "HybridQwenVLTask",
    "ImageClassificationTask",
    "KimiK3CausalLMTask",
    "KimiLinearCausalLMTask",
    "Lfm2VlTask",
    "ModelTask",
    "MllamaVisionLanguageTask",
    "MageVLTask",
    "Mistral4GGUFCausalLMTask",
    "MiniCPMVLTask",
    "MuseGlimmerVLTask",
    "MaskedDiffusionTask",
    "MoshiDepformerTask",
    "MoshiTemporalTask",
    "MiniMaxMusic3ConditionTask",
    "MiniMaxMusic3DenoisingTask",
    "MiniMaxMusic3LanguageTask",
    "MiniMaxMusic3RVQTask",
    "MiniMaxMusic3VocoderTask",
    "MultiModalTask",
    "OPSET_VERSION",
    "ObjectDetectionTask",
    "Phi4MMMultiModalTask",
    "PlamoCausalLMTask",
    "Plamo2CausalLMTask",
    "PixtralVLTask",
    "Qwen3VLVisionLanguageTask",
    "QwenImageVAETask",
    "QwenImageDenoisingTask",
    "QwenImageEditVAETask",
    "QwenImageTextEncoderTask",
    "QwenVLTask",
    "SSM2CausalLMTask",
    "SSMCausalLMTask",
    "Seq2SeqTask",
    "SpeechLanguageTask",
    "SpeechToTextTask",
    "TASK_REGISTRY",
    "TTSTask",
    "T5TextEncoderTask",
    "VAETask",
    "VideoDenoisingTask",
    "VideoVAETask",
    "VisionLanguageTask",
    "VisionEncoderDecoderTask",
    "WorldModelTask",
    "build_decoder_from_embeds",
    "build_embedding_from_features",
    "get_task",
]

from mobius._constants import OPSET_VERSION
from mobius.tasks._adapter import AdapterTask
from mobius.tasks._audio_ctc import AudioCTCTask
from mobius.tasks._audio_feature_extraction import AudioFeatureExtractionTask
from mobius.tasks._base import (
    ComponentSpec,
    ModelTask,
    build_decoder_from_embeds,
    build_embedding_from_features,
)
from mobius.tasks._causal_lm import (
    CausalLMTask,
    HybridCausalLMTask,
    SmallThinkerGGUFCausalLMTask,
)
from mobius.tasks._codec import CodecTask
from mobius.tasks._controlnet import ControlNetTask
from mobius.tasks._ctc_asr import CTCAsrTask, FeatureCTCAsrTask
from mobius.tasks._deepseek_v4 import DeepSeekV4Task
from mobius.tasks._denoising import DenoisingTask
from mobius.tasks._dflash import DFlashDraftTask
from mobius.tasks._diarization import DiarizationTask
from mobius.tasks._eagle3 import Eagle3DraftTask
from mobius.tasks._falcon_h1 import FalconH1CausalLMTask
from mobius.tasks._feature_extraction import (
    FeatureExtractionTask,
    GGUFEmbeddingFeatureExtractionTask,
    GGUFEncoderFeatureExtractionTask,
)
from mobius.tasks._fun_asr_speech_language import FunASRSpeechLanguageTask
from mobius.tasks._gguf_projector import (
    GGUFSpeakerProjectorModel,
    GGUFSpeakerProjectorTask,
    GGUFVisionProjectorModel,
    GGUFVisionProjectorTask,
)
from mobius.tasks._gemma3n import Gemma3nTask
from mobius.tasks._gemma4 import (
    Gemma4Task,
    Gemma4TextCausalLMTask,
    Gemma4UnifiedTask,
)
from mobius.tasks._gemma4_assistant import Gemma4AssistantTask
from mobius.tasks._gguf_projector import (
    GGUFSpeakerProjectorModel,
    GGUFSpeakerProjectorTask,
    GGUFVisionProjectorModel,
    GGUFVisionProjectorTask,
)
from mobius.tasks._glm_moe_dsa import GlmMoeDsaTask
from mobius.tasks._glmasr_speech_language import GlmAsrSpeechLanguageTask
from mobius.tasks._hunyuan_vl_mot import HunYuanVLMoTTask
from mobius.tasks._hy_v3_mtp import HyV3MtpTask
from mobius.tasks._image_classification import ImageClassificationTask
from mobius.tasks._kimi_k3 import KimiK3CausalLMTask
from mobius.tasks._kimi_linear import KimiLinearCausalLMTask
from mobius.tasks._masked_diffusion import MaskedDiffusionTask
from mobius.tasks._minimax_music3 import (
    MiniMaxMusic3ConditionTask,
    MiniMaxMusic3DenoisingTask,
    MiniMaxMusic3LanguageTask,
    MiniMaxMusic3RVQTask,
    MiniMaxMusic3VocoderTask,
)
from mobius.tasks._mistral4_gguf import Mistral4GGUFCausalLMTask
from mobius.tasks._moshi import MoshiDepformerTask, MoshiTemporalTask
from mobius.tasks._multimodal import MultiModalTask
from mobius.tasks._object_detection import ObjectDetectionTask
from mobius.tasks._phi4mm_multimodal import Phi4MMMultiModalTask
from mobius.tasks._plamo import PlamoCausalLMTask
from mobius.tasks._plamo2 import Plamo2CausalLMTask
from mobius.tasks._qwen4_exp import (
    Qwen4ExpCausalLMTask,
    Qwen4ExpVisionLanguageTask,
)
from mobius.tasks._qwen35_mtp import Qwen35MtpTask
from mobius.tasks._qwen_image import QwenImageDenoisingTask
from mobius.tasks._qwen_image_text_encoder import QwenImageTextEncoderTask
from mobius.tasks._qwen_image_vae import QwenImageEditVAETask, QwenImageVAETask
from mobius.tasks._rnnt import RNNTTask
from mobius.tasks._sensenova_u1 import SenseNovaU1Task
from mobius.tasks._seq2seq import Seq2SeqTask
from mobius.tasks._speech_language import SpeechLanguageTask
from mobius.tasks._speech_to_text import SpeechToTextTask
from mobius.tasks._ssm_causal_lm import SSM2CausalLMTask, SSMCausalLMTask
from mobius.tasks._t5_text_encoder import T5TextEncoderTask
from mobius.tasks._tts import TTSTask
from mobius.tasks._vae import VAETask
from mobius.tasks._video_denoising import VideoDenoisingTask
from mobius.tasks._video_vae import VideoVAETask
from mobius.tasks._vision_encoder_decoder import VisionEncoderDecoderTask
from mobius.tasks._vision_language import Qwen3VLVisionLanguageTask
from mobius.tasks._vision_language_3model import (
    Cosmos3EdgeVLTask,
    Gemma3VisionLanguageTask,
    GGUFProjectorVisionLanguageTask,
    GlmOcrVLTask,
    HybridQwenVLTask,
    Lfm2VlTask,
    MageVLTask,
    MiniCPMVLTask,
    MllamaVisionLanguageTask,
    MuseGlimmerVLTask,
    PixtralVLTask,
    QwenVLTask,
    VisionLanguageTask,
)
from mobius.tasks._world_model import WorldModelTask

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, type[ModelTask]] = {
    "adapter": AdapterTask,
    "audio-ctc": AudioCTCTask,
    "audio-feature-extraction": AudioFeatureExtractionTask,
    "ctc-asr": CTCAsrTask,
    "feature-ctc-asr": FeatureCTCAsrTask,
    "codec": CodecTask,
    "controlnet": ControlNetTask,
    "denoising": DenoisingTask,
    "diarization": DiarizationTask,
    "feature-extraction": FeatureExtractionTask,
    "gguf-encoder-feature-extraction": GGUFEncoderFeatureExtractionTask,
    "gguf-embedding-feature-extraction": GGUFEmbeddingFeatureExtractionTask,
    "masked-diffusion": MaskedDiffusionTask,
    "minimax-music3-condition": MiniMaxMusic3ConditionTask,
    "minimax-music3-denoising": MiniMaxMusic3DenoisingTask,
    "minimax-music3-language": MiniMaxMusic3LanguageTask,
    "minimax-music3-rvq": MiniMaxMusic3RVQTask,
    "minimax-music3-vocoder": MiniMaxMusic3VocoderTask,
    "image-classification": ImageClassificationTask,
    "object-detection": ObjectDetectionTask,
    "seq2seq": Seq2SeqTask,
    "moshi-depformer": MoshiDepformerTask,
    "moshi-temporal": MoshiTemporalTask,
    "text-generation": CausalLMTask,
    "smallthinker-gguf-text-generation": SmallThinkerGGUFCausalLMTask,
    "t5-text-encoding": T5TextEncoderTask,
    "deepseek-v4": DeepSeekV4Task,
    "hybrid-text-generation": HybridCausalLMTask,
    "hy-v3-mtp": HyV3MtpTask,
    "kimi-k3-text-generation": KimiK3CausalLMTask,
    "kimi-linear-text-generation": KimiLinearCausalLMTask,
    "falcon-h1-text-generation": FalconH1CausalLMTask,
    "plamo-text-generation": PlamoCausalLMTask,
    "plamo2-text-generation": Plamo2CausalLMTask,
    "dflash-draft": DFlashDraftTask,
    "eagle3-draft": Eagle3DraftTask,
    "qwen35-mtp": Qwen35MtpTask,
    "qwen4-exp-text-generation": Qwen4ExpCausalLMTask,
    "qwen4-exp-vision-language": Qwen4ExpVisionLanguageTask,
    "vae": VAETask,
    "qwen-image-vae": QwenImageVAETask,
    "qwen-image-denoising": QwenImageDenoisingTask,
    "qwen-image-edit-vae": QwenImageEditVAETask,
    "qwen-image-text-encoding": QwenImageTextEncoderTask,
    "vision-language": VisionLanguageTask,
    "gemma3-vision-language": Gemma3VisionLanguageTask,
    "vision-encoder-decoder": VisionEncoderDecoderTask,
    "cosmos3-edge-vl": Cosmos3EdgeVLTask,
    "pixtral-vl": PixtralVLTask,
    "mllama-vision-language": MllamaVisionLanguageTask,
    "mage-vl": MageVLTask,
    "muse-glimmer-vl": MuseGlimmerVLTask,
    "qwen-vl": QwenVLTask,
    "glm-ocr": GlmOcrVLTask,
    "hybrid-qwen-vl": HybridQwenVLTask,
    "minicpm-vl": MiniCPMVLTask,
    "lfm2-vl": Lfm2VlTask,
    "qwen3-vl-vision-language": Qwen3VLVisionLanguageTask,
    "gemma3n": Gemma3nTask,
    "gemma4": Gemma4Task,
    "gemma4-text-generation": Gemma4TextCausalLMTask,
    "gemma4-unified": Gemma4UnifiedTask,
    "gemma4-assistant": Gemma4AssistantTask,
    "glm-moe-dsa": GlmMoeDsaTask,
    "mistral4-gguf-text-generation": Mistral4GGUFCausalLMTask,
    "hunyuan-vl-mot": HunYuanVLMoTTask,
    "sensenova-u1": SenseNovaU1Task,
    "multimodal": MultiModalTask,
    "phi4mm-multimodal": Phi4MMMultiModalTask,
    "fun-asr-speech-language": FunASRSpeechLanguageTask,
    "glmasr-speech-language": GlmAsrSpeechLanguageTask,
    "fastconformer-rnnt": RNNTTask,
    "speech-language": SpeechLanguageTask,
    "speech-to-text": SpeechToTextTask,
    "ssm-text-generation": SSMCausalLMTask,
    "ssm2-text-generation": SSM2CausalLMTask,
    "tts": TTSTask,
    "video-denoising": VideoDenoisingTask,
    "video-vae": VideoVAETask,
    "world-model": WorldModelTask,
}


def get_task(task: str | ModelTask) -> ModelTask:
    """Resolve a task name or instance to a ModelTask.

    Args:
        task: Either a task name string (e.g. ``"text-generation"``) or
            a ``ModelTask`` instance.

    Returns:
        A ``ModelTask`` instance.

    Raises:
        ValueError: If the task name is not registered.
    """
    if isinstance(task, ModelTask):
        return task
    if task not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{task}'. Available tasks: {sorted(TASK_REGISTRY)}")
    return TASK_REGISTRY[task]()
