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
    "CTCAsrTask",
    "FeatureCTCAsrTask",
    "RNNTTask",
    "CodecTask",
    "ComponentSpec",
    "ControlNetTask",
    "Cosmos3AVAEAudioDecoderTask",
    "Cosmos3AVAEAudioTokenizerTask",
    "Cosmos3OmniGeneratorTask",
    "DeepSeekV4Task",
    "DFlashDraftTask",
    "Eagle3DraftTask",
    "Qwen35MtpTask",
    "DenoisingTask",
    "DiarizationTask",
    "FeatureExtractionTask",
    "FunASRSpeechLanguageTask",
    "Gemma3nTask",
    "Gemma4AssistantTask",
    "Gemma4Task",
    "Gemma4UnifiedTask",
    "Gemma4TextCausalLMTask",
    "HybridCausalLMTask",
    "Cosmos3EdgeVLTask",
    "HybridQwenVLTask",
    "ImageClassificationTask",
    "LatentDynamicsTask",
    "ModelTask",
    "MllamaVisionLanguageTask",
    "MageVLTask",
    "MiniCPMVLTask",
    "MuseGlimmerVLTask",
    "MaskedDiffusionTask",
    "MoshiDepformerTask",
    "MoshiTemporalTask",
    "MultiModalTask",
    "OPSET_VERSION",
    "ObjectDetectionTask",
    "Phi4MMMultiModalTask",
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
    "VAETask",
    "VideoDenoisingTask",
    "VisionLanguageTask",
    "VisionEncoderDecoderTask",
    "WorldModelTask",
    "WanVAETask",
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
)
from mobius.tasks._codec import CodecTask
from mobius.tasks._controlnet import ControlNetTask
from mobius.tasks._cosmos3_audio import (
    Cosmos3AVAEAudioDecoderTask,
    Cosmos3AVAEAudioTokenizerTask,
)
from mobius.tasks._cosmos3_omni_generator import Cosmos3OmniGeneratorTask
from mobius.tasks._ctc_asr import CTCAsrTask, FeatureCTCAsrTask
from mobius.tasks._deepseek_v4 import DeepSeekV4Task
from mobius.tasks._denoising import DenoisingTask
from mobius.tasks._dflash import DFlashDraftTask
from mobius.tasks._diarization import DiarizationTask
from mobius.tasks._eagle3 import Eagle3DraftTask
from mobius.tasks._feature_extraction import FeatureExtractionTask
from mobius.tasks._fun_asr_speech_language import FunASRSpeechLanguageTask
from mobius.tasks._gemma3n import Gemma3nTask
from mobius.tasks._gemma4 import (
    Gemma4Task,
    Gemma4TextCausalLMTask,
    Gemma4UnifiedTask,
)
from mobius.tasks._gemma4_assistant import Gemma4AssistantTask
from mobius.tasks._hunyuan_vl_mot import HunYuanVLMoTTask
from mobius.tasks._image_classification import ImageClassificationTask
from mobius.tasks._masked_diffusion import MaskedDiffusionTask
from mobius.tasks._moshi import MoshiDepformerTask, MoshiTemporalTask
from mobius.tasks._multimodal import MultiModalTask
from mobius.tasks._object_detection import ObjectDetectionTask
from mobius.tasks._phi4mm_multimodal import Phi4MMMultiModalTask
from mobius.tasks._qwen35_mtp import Qwen35MtpTask
from mobius.tasks._qwen_image import QwenImageDenoisingTask
from mobius.tasks._qwen_image_text_encoder import QwenImageTextEncoderTask
from mobius.tasks._qwen_image_vae import QwenImageEditVAETask, QwenImageVAETask
from mobius.tasks._rnnt import RNNTTask
from mobius.tasks._seq2seq import Seq2SeqTask
from mobius.tasks._speech_language import SpeechLanguageTask
from mobius.tasks._speech_to_text import SpeechToTextTask
from mobius.tasks._ssm_causal_lm import SSM2CausalLMTask, SSMCausalLMTask
from mobius.tasks._tts import TTSTask
from mobius.tasks._vae import VAETask
from mobius.tasks._video_denoising import VideoDenoisingTask
from mobius.tasks._vision_encoder_decoder import VisionEncoderDecoderTask
from mobius.tasks._vision_language import Qwen3VLVisionLanguageTask
from mobius.tasks._vision_language_3model import (
    Cosmos3EdgeVLTask,
    HybridQwenVLTask,
    MageVLTask,
    MiniCPMVLTask,
    MllamaVisionLanguageTask,
    MuseGlimmerVLTask,
    PixtralVLTask,
    QwenVLTask,
    VisionLanguageTask,
)
from mobius.tasks._wan_vae import WanVAETask
from mobius.tasks._world_model import LatentDynamicsTask, WorldModelTask

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
    "cosmos3-audio-decoder": Cosmos3AVAEAudioDecoderTask,
    "cosmos3-audio-tokenizer": Cosmos3AVAEAudioTokenizerTask,
    "cosmos3-omni-generator": Cosmos3OmniGeneratorTask,
    "denoising": DenoisingTask,
    "diarization": DiarizationTask,
    "feature-extraction": FeatureExtractionTask,
    "masked-diffusion": MaskedDiffusionTask,
    "image-classification": ImageClassificationTask,
    "object-detection": ObjectDetectionTask,
    "seq2seq": Seq2SeqTask,
    "moshi-depformer": MoshiDepformerTask,
    "moshi-temporal": MoshiTemporalTask,
    "text-generation": CausalLMTask,
    "deepseek-v4": DeepSeekV4Task,
    "hybrid-text-generation": HybridCausalLMTask,
    "dflash-draft": DFlashDraftTask,
    "eagle3-draft": Eagle3DraftTask,
    "qwen35-mtp": Qwen35MtpTask,
    "vae": VAETask,
    "wan-vae": WanVAETask,
    "qwen-image-vae": QwenImageVAETask,
    "qwen-image-denoising": QwenImageDenoisingTask,
    "qwen-image-edit-vae": QwenImageEditVAETask,
    "qwen-image-text-encoding": QwenImageTextEncoderTask,
    "vision-language": VisionLanguageTask,
    "vision-encoder-decoder": VisionEncoderDecoderTask,
    "cosmos3-edge-vl": Cosmos3EdgeVLTask,
    "pixtral-vl": PixtralVLTask,
    "mllama-vision-language": MllamaVisionLanguageTask,
    "mage-vl": MageVLTask,
    "muse-glimmer-vl": MuseGlimmerVLTask,
    "qwen-vl": QwenVLTask,
    "hybrid-qwen-vl": HybridQwenVLTask,
    "minicpm-vl": MiniCPMVLTask,
    "qwen3-vl-vision-language": Qwen3VLVisionLanguageTask,
    "gemma3n": Gemma3nTask,
    "gemma4": Gemma4Task,
    "gemma4-text-generation": Gemma4TextCausalLMTask,
    "gemma4-unified": Gemma4UnifiedTask,
    "gemma4-assistant": Gemma4AssistantTask,
    "hunyuan-vl-mot": HunYuanVLMoTTask,
    "multimodal": MultiModalTask,
    "phi4mm-multimodal": Phi4MMMultiModalTask,
    "fun-asr-speech-language": FunASRSpeechLanguageTask,
    "fastconformer-rnnt": RNNTTask,
    "speech-language": SpeechLanguageTask,
    "speech-to-text": SpeechToTextTask,
    "ssm-text-generation": SSMCausalLMTask,
    "ssm2-text-generation": SSM2CausalLMTask,
    "tts": TTSTask,
    "video-denoising": VideoDenoisingTask,
    "latent-dynamics": LatentDynamicsTask,
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
