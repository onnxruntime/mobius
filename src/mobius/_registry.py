# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Model registry mapping architecture names to module classes.

The registry is the central lookup table that maps HuggingFace model
architecture names (``config.model_type``) to the ``nn.Module`` subclass,
default task, and config class used to build the ONNX graph.
"""

from __future__ import annotations

__all__ = [
    "MODEL_MAP",
    "ModelRegistration",
    "ModelRegistry",
    "registry",
]

import dataclasses
import difflib

from onnxscript import nn

from mobius._configs import (
    BaseModelConfig,
    Gemma4Config,
    WhisperConfig,
)
from mobius.models import (
    ApertusCausalLMModel,
    ArceeCausalLMModel,
    CausalLMModel,
    ChatGLMCausalLMModel,
    DeepSeekOCR2CausalLMModel,
    DeepSeekV3CausalLMModel,
    DiffLlamaCausalLMModel,
    DogeCausalLMModel,
    Ernie45MoECausalLMModel,
    ErnieCausalLMModel,
    ExaOne4CausalLMModel,
    Gemma2CausalLMModel,
    Gemma3CausalLMModel,
    Gemma3MultiModalModel,
    Gemma4CausalLMModel,
    Gemma4Model,
    GemmaCausalLMModel,
    Glm4CausalLMModel,
    Glm4MoECausalLMModel,
    GlmCausalLMModel,
    GPTOSSCausalLMModel,
    GraniteCausalLMModel,
    GraniteMoECausalLMModel,
    HunYuanMoEV1CausalLMModel,
    HunYuanV1DenseCausalLMModel,
    InternLM2CausalLMModel,
    LayerNormCausalLMModel,
    Llama4CausalLMModel,
    MoECausalLMModel,
    NanoChatCausalLMModel,
    NemotronCausalLMModel,
    OLMo2CausalLMModel,
    OLMoCausalLMModel,
    Phi3CausalLMModel,
    Phi3MoECausalLMModel,
    Phi3SmallCausalLMModel,
    Phi4MMMultiModalModel,
    PhiCausalLMModel,
    Qwen2MoECausalLMModel,
    Qwen2VLCausalLMModel,
    Qwen3CausalLMModel,
    Qwen3NextCausalLMModel,
    Qwen3VL3ModelCausalLMModel,
    Qwen3VLCausalLMModel,
    Qwen3VLTextModel,
    Qwen25VLCausalLMModel,
    Qwen25VLTextModel,
    Qwen35CausalLMModel,
    Qwen35MoECausalLMModel,
    Qwen35VL3ModelCausalLMModel,
    Qwen35VLTextModel,
    QwenCausalLMModel,
    SmolLM3CausalLMModel,
    WhisperForConditionalGeneration,
)
from mobius.models.bamba import BambaCausalLMModel
from mobius.models.bart import BartForConditionalGeneration
from mobius.models.bert import BertModel
from mobius.models.blip import BlipVisionModel
from mobius.models.blip2 import Blip2Model
from mobius.models.clip import CLIPTextModel, CLIPVisionModel, SigLIPVisionModel
from mobius.models.cohere import CohereCausalLMModel
from mobius.models.ctrl import CTRLCausalLMModel
from mobius.models.depth_anything import DepthAnythingForDepthEstimation
from mobius.models.distilbert import DistilBertModel
from mobius.models.falcon import BloomCausalLMModel, FalconCausalLMModel, MPTCausalLMModel
from mobius.models.fun_asr import FunASRForConditionalGeneration
from mobius.models.gemma3n import Gemma3nCausalLMModel
from mobius.models.gpt2 import GPT2CausalLMModel
from mobius.models.gpt_neox import GPTNeoXCausalLMModel, GPTNeoXJapaneseCausalLMModel
from mobius.models.gptj_codegen import CodeGenCausalLMModel, GPTJCausalLMModel
from mobius.models.granitemoehybrid import GraniteMoeHybridCausalLMModel
from mobius.models.internvl import InternVL2Model
from mobius.models.jamba import JambaCausalLMModel
from mobius.models.jetmoe import JetMoeCausalLMModel
from mobius.models.layoutlmv3 import LayoutLMv3Model
from mobius.models.llava import LLaVAModel
from mobius.models.longcat_flash import LongcatFlashCausalLMModel
from mobius.models.mamba import Mamba2CausalLMModel, MambaCausalLMModel
from mobius.models.minimax import MiniMaxCausalLMModel
from mobius.models.mllama import MllamaCausalLMModel
from mobius.models.modernbert import ModernBertDecoderModel, ModernBertModel
from mobius.models.nemotron_h import NemotronHCausalLMModel
from mobius.models.opt import OPTCausalLMModel
from mobius.models.persimmon import PersimmonCausalLMModel
from mobius.models.qwen3_asr import Qwen3ASRForConditionalGeneration
from mobius.models.qwen3_tts import Qwen3TTSForConditionalGeneration
from mobius.models.qwen3_tts_tokenizer import Qwen3TTSTokenizerV2Model
from mobius.models.sam2 import Sam2VisionModel
from mobius.models.segformer import SegformerForSemanticSegmentation
from mobius.models.sensevoice_small import SenseVoiceSmallModel
from mobius.models.starcoder2 import StarCoder2CausalLMModel
from mobius.models.t5 import T5ForConditionalGeneration
from mobius.models.trocr import TrOCRForConditionalGeneration
from mobius.models.vit import ViTModel
from mobius.models.wav2vec2 import Wav2Vec2Model
from mobius.models.xlm import XLMCausalLMModel
from mobius.models.yolos import YolosForObjectDetection
from mobius.models.zamba2 import Zamba2CausalLMModel


@dataclasses.dataclass(frozen=True)
class ModelRegistration:
    """A single entry in the model registry.

    Attributes:
        module_class: The ``nn.Module`` subclass that builds the ONNX graph.
        task: Default task name (e.g. ``"text-generation"``).  When ``None``
            the task is read from ``module_class.default_task`` at resolution time.
        config_class: Config class for parsing HuggingFace configs.  When ``None``
            the class is read from ``module_class.config_class`` at resolution time.
        test_model_id: HuggingFace model ID used for L2 architecture validation.
            The config.json is downloaded (no weights) to verify that the real-size
            ONNX graph can be built.  ``None`` means no L2 test is defined.
        family: Dashboard family grouping (e.g. ``"phi"`` for phi3, phi3small,
            phimoe).  ``None`` means auto-derive from the model_type prefix.
        variant: Short label identifying the code-path variant (e.g. ``"mla"``,
            ``"moe"``, ``"sliding_window"``).  Used for dashboard display.
    """

    module_class: type[nn.Module]
    task: str | None = None
    config_class: type[BaseModelConfig] | None = None
    test_model_id: str | None = None
    family: str | None = None
    variant: str | None = None


class ModelRegistry:
    """Registry mapping architecture names to module classes, tasks, and configs.

    The registry is used by :func:`build` to auto-detect the module class,
    default task, and config class for a given HuggingFace model.  Users can
    register custom architectures::

        from mobius import registry

        # Simple (module class only — backward compatible)
        registry.register("my_arch", MyModelClass)

        # Full (module + task + config)
        registry.register(
            "my_arch", MyModelClass,
            task="text-generation",
            config_class=MyConfig,
        )
    """

    def __init__(self) -> None:
        self._map: dict[str, ModelRegistration] = {}

    def register(
        self,
        architecture: str,
        module_class: type[nn.Module],
        *,
        task: str | None = None,
        config_class: type[BaseModelConfig] | None = None,
        test_model_id: str | None = None,
        family: str | None = None,
        variant: str | None = None,
    ) -> None:
        """Register a module class for an architecture name.

        Args:
            architecture: The architecture name (matching HF ``config.model_type``).
            module_class: The module class to use for this architecture.
            task: Default task name for this architecture. When ``None``,
                the task is read from ``module_class.default_task``.
            config_class: Config class for this architecture. When ``None``,
                the class is read from ``module_class.config_class``.
            test_model_id: HuggingFace model ID for L2 architecture validation.
            family: Dashboard family grouping override.
            variant: Short label for the code-path variant.
        """
        self._map[architecture] = ModelRegistration(
            module_class,
            task,
            config_class,
            test_model_id,
            family,
            variant,
        )

    def get(self, architecture: str) -> type[nn.Module]:
        """Look up the module class for an architecture.

        Args:
            architecture: The architecture name.

        Returns:
            The registered module class.

        Raises:
            KeyError: If the architecture is not registered.
        """
        if architecture not in self._map:
            raise KeyError(self._not_found_message(architecture))
        return self._map[architecture].module_class

    def get_registration(self, architecture: str) -> ModelRegistration:
        """Look up the full registration entry for an architecture.

        Args:
            architecture: The architecture name.

        Returns:
            The :class:`ModelRegistration` entry.

        Raises:
            KeyError: If the architecture is not registered.
        """
        if architecture not in self._map:
            raise KeyError(self._not_found_message(architecture))
        return self._map[architecture]

    def _not_found_message(self, architecture: str) -> str:
        """Build a helpful error message for unknown architectures."""
        suggestions = difflib.get_close_matches(
            architecture, self._map.keys(), n=3, cutoff=0.6
        )
        msg = f"Unknown model_type '{architecture}'."
        if suggestions:
            quoted = ", ".join(f"'{s}'" for s in suggestions)
            msg += f" Did you mean: {quoted}?"
        msg += f" Use registry.register('{architecture}', YourModuleClass) to add it."
        return msg

    def get_task(self, architecture: str) -> str | None:
        """Return the registered default task, or ``None``."""
        return self._map[architecture].task if architecture in self._map else None

    def get_config_class(self, architecture: str) -> type[BaseModelConfig] | None:
        """Return the registered config class, or ``None``."""
        return self._map[architecture].config_class if architecture in self._map else None

    def __contains__(self, architecture: str) -> bool:
        return architecture in self._map

    def __len__(self) -> int:
        return len(self._map)

    def architectures(self) -> list[str]:
        """Return a sorted list of registered architecture names."""
        return sorted(self._map)


def _detect_fallback_registration(hf_config) -> ModelRegistration | None:
    """Detect a compatible model class for an unregistered model type.

    Analyzes a HuggingFace config to determine if the model is
    architecturally compatible with a built-in base class.  This enables
    automatic support for new Llama-like or MoE decoder-only models
    without explicit registration.

    Only returns a fallback when the config clearly indicates a standard
    causal-LM transformer.  Composite models (multimodal, speech),
    encoder-decoder, and encoder-only architectures return ``None``.

    Args:
        hf_config: A HuggingFace ``PretrainedConfig`` (or compatible object).

    Returns:
        A :class:`ModelRegistration` if a compatible fallback is found,
        or ``None`` otherwise.
    """
    # Reject encoder-decoder models — too varied for auto-fallback
    if getattr(hf_config, "is_encoder_decoder", False):
        return None

    # Reject composite models that need custom encoders/projectors
    if hasattr(hf_config, "vision_config") or hasattr(hf_config, "audio_config"):
        return None

    # Reject SSM/recurrent models — they have CausalLM in architectures
    # but use fundamentally different computation (not transformer attention)
    _ssm_indicators = (
        "d_state",
        "d_conv",
        "ssm_cfg",
        "recurrent_block_type",
        "rescale_every",  # RWKV linear-attention models
    )
    if any(getattr(hf_config, attr, None) is not None for attr in _ssm_indicators):
        return None

    # Check HF architectures field for causal LM indicator
    architectures = getattr(hf_config, "architectures", None) or []
    if not any("CausalLM" in arch for arch in architectures):
        return None

    # Require minimum config fields for graph construction
    required_fields = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "vocab_size",
    )
    if not all(getattr(hf_config, f, 0) > 0 for f in required_fields):
        return None

    # MoE detection: route to MoE class if experts are present
    num_experts = getattr(hf_config, "num_local_experts", 0)
    if num_experts and num_experts > 1:
        return ModelRegistration(MoECausalLMModel)

    return ModelRegistration(CausalLMModel)


# ---------------------------------------------------------------------------
# Declarative registration table
#
# Each entry maps an architecture name (HF ``config.model_type``) to a
# ``ModelRegistration``.  Use keyword arguments when ``task`` or
# ``config_class`` differ from their defaults (``None``).
#
# test_model_id, family, and variant are applied separately via
# ``_apply_test_metadata()`` using the dicts below.
# ---------------------------------------------------------------------------
_REGISTRATIONS: dict[str, ModelRegistration] = {
    # --- Text Generation (Llama-compatible) ---
    "baichuan": ModelRegistration(CausalLMModel),
    "code_llama": ModelRegistration(CausalLMModel),
    "codegen2": ModelRegistration(CausalLMModel),
    "command_r": ModelRegistration(CausalLMModel),
    "csm": ModelRegistration(CausalLMModel),
    "dots1": ModelRegistration(DeepSeekV3CausalLMModel),
    "evolla": ModelRegistration(CausalLMModel),
    "exaone": ModelRegistration(CausalLMModel),
    "helium": ModelRegistration(CausalLMModel),
    "llama": ModelRegistration(CausalLMModel),
    "minicpm": ModelRegistration(CausalLMModel),
    "minicpm3": ModelRegistration(CausalLMModel),
    "ministral": ModelRegistration(CausalLMModel),
    "ministral3": ModelRegistration(CausalLMModel),
    "mistral": ModelRegistration(CausalLMModel),
    "open-llama": ModelRegistration(CausalLMModel),
    "openelm": ModelRegistration(CausalLMModel),
    "qwen2": ModelRegistration(CausalLMModel),
    "seed_oss": ModelRegistration(CausalLMModel),
    "solar_open": ModelRegistration(CausalLMModel),
    "yi": ModelRegistration(CausalLMModel),
    "youtu": ModelRegistration(DeepSeekV3CausalLMModel),
    "zamba": ModelRegistration(CausalLMModel),
    "zamba2": ModelRegistration(Zamba2CausalLMModel),
    # --- Text Generation (architecture-specific) ---
    "apertus": ModelRegistration(ApertusCausalLMModel),
    "arcee": ModelRegistration(ArceeCausalLMModel),
    "bloom": ModelRegistration(BloomCausalLMModel),
    "chatglm": ModelRegistration(ChatGLMCausalLMModel),
    "codegen": ModelRegistration(CodeGenCausalLMModel),
    "cohere": ModelRegistration(CohereCausalLMModel),
    "cohere2": ModelRegistration(CohereCausalLMModel),
    "diffllama": ModelRegistration(DiffLlamaCausalLMModel),
    "doge": ModelRegistration(DogeCausalLMModel),
    "ernie4_5": ModelRegistration(ErnieCausalLMModel),
    "exaone4": ModelRegistration(ExaOne4CausalLMModel),
    "falcon": ModelRegistration(FalconCausalLMModel),
    "falcon_h1": ModelRegistration(FalconCausalLMModel),
    "gemma": ModelRegistration(GemmaCausalLMModel),
    "gemma2": ModelRegistration(Gemma2CausalLMModel),
    "gemma3": ModelRegistration(Gemma3MultiModalModel, task="vision-language"),
    "gemma3_text": ModelRegistration(Gemma3CausalLMModel),
    "gemma3n": ModelRegistration(Gemma3nCausalLMModel),
    "gemma3n_text": ModelRegistration(Gemma3nCausalLMModel),
    "gemma4_text": ModelRegistration(Gemma4CausalLMModel, config_class=Gemma4Config),
    "glm": ModelRegistration(GlmCausalLMModel),
    "glm4": ModelRegistration(Glm4CausalLMModel),
    "gpt_neox": ModelRegistration(GPTNeoXCausalLMModel),
    "gpt_neox_japanese": ModelRegistration(GPTNeoXJapaneseCausalLMModel),
    "gpt_oss": ModelRegistration(GPTOSSCausalLMModel),
    "gptj": ModelRegistration(GPTJCausalLMModel),
    "granite": ModelRegistration(GraniteCausalLMModel),
    "hunyuan_v1_dense": ModelRegistration(HunYuanV1DenseCausalLMModel),
    "internlm2": ModelRegistration(InternLM2CausalLMModel),
    "llama4_text": ModelRegistration(Llama4CausalLMModel),
    "modernbert-decoder": ModelRegistration(ModernBertDecoderModel),
    "mpt": ModelRegistration(MPTCausalLMModel),
    "nanochat": ModelRegistration(NanoChatCausalLMModel),
    "nemotron": ModelRegistration(NemotronCausalLMModel),
    "olmo": ModelRegistration(OLMoCausalLMModel),
    "olmo2": ModelRegistration(OLMo2CausalLMModel),
    "olmo3": ModelRegistration(OLMo2CausalLMModel),
    "persimmon": ModelRegistration(PersimmonCausalLMModel),
    "phi": ModelRegistration(PhiCausalLMModel),
    "phi3": ModelRegistration(Phi3CausalLMModel),
    "phi3small": ModelRegistration(Phi3SmallCausalLMModel),
    "qwen": ModelRegistration(QwenCausalLMModel),
    "qwen3": ModelRegistration(Qwen3CausalLMModel),
    "qwen3_5_text": ModelRegistration(Qwen35CausalLMModel),
    "shieldgemma2": ModelRegistration(Gemma2CausalLMModel),
    "smollm3": ModelRegistration(SmolLM3CausalLMModel),
    "stablelm": ModelRegistration(LayerNormCausalLMModel),
    "starcoder2": ModelRegistration(StarCoder2CausalLMModel),
    # --- Mixture of Experts ---
    "arctic": ModelRegistration(MoECausalLMModel),
    "dbrx": ModelRegistration(MoECausalLMModel),
    "ernie4_5_moe": ModelRegistration(Ernie45MoECausalLMModel),
    "flex_olmo": ModelRegistration(MoECausalLMModel),
    "glm4_moe": ModelRegistration(Glm4MoECausalLMModel),
    "granitemoe": ModelRegistration(GraniteMoECausalLMModel),
    "granitemoehybrid": ModelRegistration(GraniteMoeHybridCausalLMModel),
    "granitemoeshared": ModelRegistration(GraniteMoECausalLMModel),
    "hunyuan_v1_moe": ModelRegistration(HunYuanMoEV1CausalLMModel),
    "jetmoe": ModelRegistration(JetMoeCausalLMModel),
    "minimax": ModelRegistration(MiniMaxCausalLMModel),
    "mixtral": ModelRegistration(MoECausalLMModel),
    "olmoe": ModelRegistration(MoECausalLMModel),
    "phimoe": ModelRegistration(Phi3MoECausalLMModel),
    "qwen2_moe": ModelRegistration(Qwen2MoECausalLMModel),
    "qwen3_5_moe": ModelRegistration(Qwen35MoECausalLMModel),
    "qwen3_moe": ModelRegistration(MoECausalLMModel),
    "qwen3_next": ModelRegistration(Qwen3NextCausalLMModel),
    "qwen3_omni_moe": ModelRegistration(MoECausalLMModel),
    "qwen3_vl_moe": ModelRegistration(MoECausalLMModel),
    # --- DeepSeek (MLA + MoE) ---
    "deepseek_v2": ModelRegistration(DeepSeekV3CausalLMModel),
    "deepseek_v2_moe": ModelRegistration(DeepSeekV3CausalLMModel),
    "deepseek_v3": ModelRegistration(DeepSeekV3CausalLMModel),
    "deepseek_vl_v2": ModelRegistration(DeepSeekOCR2CausalLMModel),
    # --- SSM (Mamba / Mamba2) ---
    "falcon_mamba": ModelRegistration(MambaCausalLMModel),
    "mamba": ModelRegistration(MambaCausalLMModel),
    "mamba2": ModelRegistration(Mamba2CausalLMModel),
    # --- Hybrid SSM+Attention ---
    "bamba": ModelRegistration(BambaCausalLMModel),
    "jamba": ModelRegistration(JambaCausalLMModel),
    "nemotron_h": ModelRegistration(NemotronHCausalLMModel),
    # --- Hybrid linear-attention ---
    "longcat_flash": ModelRegistration(LongcatFlashCausalLMModel),
    # --- Multimodal ---
    "aya_vision": ModelRegistration(LLaVAModel, task="vision-language"),
    "blip-2": ModelRegistration(Blip2Model, task="vision-language"),
    "chameleon": ModelRegistration(LLaVAModel, task="vision-language"),
    "cohere2_vision": ModelRegistration(LLaVAModel, task="vision-language"),
    "deepseek_vl": ModelRegistration(LLaVAModel, task="vision-language"),
    "deepseek_vl_hybrid": ModelRegistration(LLaVAModel, task="vision-language"),
    "florence2": ModelRegistration(LLaVAModel, task="vision-language"),
    "fuyu": ModelRegistration(LLaVAModel, task="vision-language"),
    "gemma4": ModelRegistration(Gemma4Model, task="gemma4", config_class=Gemma4Config),
    "glm4v": ModelRegistration(LLaVAModel, task="vision-language"),
    "glm4v_moe": ModelRegistration(LLaVAModel, task="vision-language"),
    "glm4v_moe_text": ModelRegistration(Glm4MoECausalLMModel),
    "glm4v_text": ModelRegistration(Glm4CausalLMModel),
    "got_ocr2": ModelRegistration(LLaVAModel, task="vision-language"),
    "idefics2": ModelRegistration(LLaVAModel, task="vision-language"),
    "idefics3": ModelRegistration(LLaVAModel, task="vision-language"),
    "instructblip": ModelRegistration(LLaVAModel, task="vision-language"),
    "instructblipvideo": ModelRegistration(LLaVAModel, task="vision-language"),
    "internvl": ModelRegistration(InternVL2Model, task="vision-language"),
    "internvl2": ModelRegistration(InternVL2Model, task="vision-language"),
    "internvl_chat": ModelRegistration(InternVL2Model, task="vision-language"),
    "janus": ModelRegistration(LLaVAModel, task="vision-language"),
    "llava": ModelRegistration(LLaVAModel, task="vision-language"),
    "llava_next": ModelRegistration(LLaVAModel, task="vision-language"),
    "llava_next_video": ModelRegistration(LLaVAModel, task="vision-language"),
    "llava_onevision": ModelRegistration(LLaVAModel, task="vision-language"),
    "mistral3": ModelRegistration(LLaVAModel, task="pixtral-vl"),
    "mllama": ModelRegistration(MllamaCausalLMModel, task="mllama-vision-language"),
    "molmo": ModelRegistration(LLaVAModel, task="vision-language"),
    "ovis2": ModelRegistration(LLaVAModel, task="vision-language"),
    "paligemma": ModelRegistration(LLaVAModel, task="vision-language"),
    "phi4_multimodal": ModelRegistration(Phi4MMMultiModalModel, task="phi4mm-multimodal"),
    "phi4mm": ModelRegistration(Phi4MMMultiModalModel, task="phi4mm-multimodal"),
    "pixtral": ModelRegistration(LLaVAModel, task="pixtral-vl"),
    "qwen2_5_vl": ModelRegistration(Qwen25VLCausalLMModel, task="qwen-vl"),
    "qwen2_5_vl_text": ModelRegistration(Qwen25VLTextModel),
    "qwen2_vl": ModelRegistration(Qwen2VLCausalLMModel, task="qwen-vl"),
    "qwen2_vl_text": ModelRegistration(Qwen25VLTextModel),
    "qwen3_5": ModelRegistration(Qwen35VL3ModelCausalLMModel, task="hybrid-qwen-vl"),
    "qwen3_5_vl": ModelRegistration(Qwen35VL3ModelCausalLMModel, task="hybrid-qwen-vl"),
    "qwen3_5_vl_text": ModelRegistration(Qwen35VLTextModel),
    "qwen3_vl": ModelRegistration(Qwen3VL3ModelCausalLMModel, task="qwen-vl"),
    "qwen3_vl_single": ModelRegistration(
        Qwen3VLCausalLMModel, task="qwen3-vl-vision-language"
    ),
    "qwen3_vl_text": ModelRegistration(Qwen3VLTextModel),
    "smolvlm": ModelRegistration(LLaVAModel, task="vision-language"),
    "video_llava": ModelRegistration(LLaVAModel, task="vision-language"),
    "vipllava": ModelRegistration(LLaVAModel, task="vision-language"),
    # --- Speech ---
    "fun_asr": ModelRegistration(
        FunASRForConditionalGeneration, task="fun-asr-speech-language"
    ),
    "qwen3_asr": ModelRegistration(Qwen3ASRForConditionalGeneration, task="speech-language"),
    "qwen3_forced_aligner": ModelRegistration(
        Qwen3ASRForConditionalGeneration, task="speech-language"
    ),
    "sensevoice_small": ModelRegistration(SenseVoiceSmallModel, task="audio-ctc"),
    "qwen3_tts": ModelRegistration(Qwen3TTSForConditionalGeneration),
    "qwen3_tts_tokenizer_12hz": ModelRegistration(Qwen3TTSTokenizerV2Model, task="codec"),
    "whisper": ModelRegistration(
        WhisperForConditionalGeneration,
        task="speech-to-text",
        config_class=WhisperConfig,
    ),
    # --- Encoder-only ---
    "albert": ModelRegistration(BertModel, task="feature-extraction"),
    "bert": ModelRegistration(BertModel, task="feature-extraction"),
    "bros": ModelRegistration(BertModel, task="feature-extraction"),
    "camembert": ModelRegistration(BertModel, task="feature-extraction"),
    "clip_text_model": ModelRegistration(CLIPTextModel, task="feature-extraction"),
    "data2vec-text": ModelRegistration(BertModel, task="feature-extraction"),
    "deberta": ModelRegistration(BertModel, task="feature-extraction"),
    "deberta-v2": ModelRegistration(BertModel, task="feature-extraction"),
    "distilbert": ModelRegistration(DistilBertModel, task="feature-extraction"),
    "electra": ModelRegistration(BertModel, task="feature-extraction"),
    "ernie": ModelRegistration(BertModel, task="feature-extraction"),
    "ernie_m": ModelRegistration(BertModel, task="feature-extraction"),
    "esm": ModelRegistration(BertModel, task="feature-extraction"),
    "flaubert": ModelRegistration(BertModel, task="feature-extraction"),
    "ibert": ModelRegistration(BertModel, task="feature-extraction"),
    "layoutlm": ModelRegistration(BertModel, task="feature-extraction"),
    "layoutlmv2": ModelRegistration(BertModel, task="feature-extraction"),
    "layoutlmv3": ModelRegistration(LayoutLMv3Model, task="feature-extraction"),
    "lilt": ModelRegistration(BertModel, task="feature-extraction"),
    "markuplm": ModelRegistration(BertModel, task="feature-extraction"),
    "mega": ModelRegistration(BertModel, task="feature-extraction"),
    "megatron-bert": ModelRegistration(BertModel, task="feature-extraction"),
    "mobilebert": ModelRegistration(BertModel, task="feature-extraction"),
    "modernbert": ModelRegistration(ModernBertModel, task="feature-extraction"),
    "mpnet": ModelRegistration(BertModel, task="feature-extraction"),
    "mra": ModelRegistration(BertModel, task="feature-extraction"),
    "nezha": ModelRegistration(BertModel, task="feature-extraction"),
    "nystromformer": ModelRegistration(BertModel, task="feature-extraction"),
    "qdqbert": ModelRegistration(BertModel, task="feature-extraction"),
    "rembert": ModelRegistration(BertModel, task="feature-extraction"),
    "roberta": ModelRegistration(BertModel, task="feature-extraction"),
    "roberta-prelayernorm": ModelRegistration(BertModel, task="feature-extraction"),
    "roc_bert": ModelRegistration(BertModel, task="feature-extraction"),
    "roformer": ModelRegistration(BertModel, task="feature-extraction"),
    "splinter": ModelRegistration(BertModel, task="feature-extraction"),
    "squeezebert": ModelRegistration(BertModel, task="feature-extraction"),
    "xlm-roberta": ModelRegistration(BertModel, task="feature-extraction"),
    "xlm-roberta-xl": ModelRegistration(BertModel, task="feature-extraction"),
    "xlnet": ModelRegistration(BertModel, task="feature-extraction"),
    "xmod": ModelRegistration(BertModel, task="feature-extraction"),
    "yoso": ModelRegistration(BertModel, task="feature-extraction"),
    # --- Absolute positional embeddings (non-RoPE) ---
    "biogpt": ModelRegistration(GPT2CausalLMModel),
    "ctrl": ModelRegistration(CTRLCausalLMModel),
    "gpt-sw3": ModelRegistration(GPT2CausalLMModel),
    "gpt2": ModelRegistration(GPT2CausalLMModel),
    "gpt_bigcode": ModelRegistration(GPT2CausalLMModel),
    "gpt_neo": ModelRegistration(GPT2CausalLMModel),
    "imagegpt": ModelRegistration(GPT2CausalLMModel),
    "openai-gpt": ModelRegistration(GPT2CausalLMModel),
    "opt": ModelRegistration(OPTCausalLMModel),
    "xglm": ModelRegistration(GPT2CausalLMModel),
    "xlm": ModelRegistration(XLMCausalLMModel),
    # --- Encoder-decoder ---
    "bart": ModelRegistration(BartForConditionalGeneration, task="seq2seq"),
    "bigbird_pegasus": ModelRegistration(BartForConditionalGeneration, task="seq2seq"),
    "blenderbot": ModelRegistration(BartForConditionalGeneration, task="seq2seq"),
    "blenderbot-small": ModelRegistration(BartForConditionalGeneration, task="seq2seq"),
    "fsmt": ModelRegistration(BartForConditionalGeneration, task="seq2seq"),
    "led": ModelRegistration(BartForConditionalGeneration, task="seq2seq"),
    "longt5": ModelRegistration(T5ForConditionalGeneration, task="seq2seq"),
    "m2m_100": ModelRegistration(BartForConditionalGeneration, task="seq2seq"),
    "marian": ModelRegistration(BartForConditionalGeneration, task="seq2seq"),
    "mbart": ModelRegistration(BartForConditionalGeneration, task="seq2seq"),
    "mt5": ModelRegistration(T5ForConditionalGeneration, task="seq2seq"),
    "mvp": ModelRegistration(BartForConditionalGeneration, task="seq2seq"),
    "nllb-moe": ModelRegistration(BartForConditionalGeneration, task="seq2seq"),
    "nllb_moe": ModelRegistration(BartForConditionalGeneration, task="seq2seq"),
    "pegasus": ModelRegistration(BartForConditionalGeneration, task="seq2seq"),
    "pegasus_x": ModelRegistration(BartForConditionalGeneration, task="seq2seq"),
    "plbart": ModelRegistration(BartForConditionalGeneration, task="seq2seq"),
    "prophetnet": ModelRegistration(BartForConditionalGeneration, task="seq2seq"),
    "switch_transformers": ModelRegistration(T5ForConditionalGeneration, task="seq2seq"),
    "t5": ModelRegistration(T5ForConditionalGeneration, task="seq2seq"),
    "trocr": ModelRegistration(TrOCRForConditionalGeneration, task="seq2seq"),
    "umt5": ModelRegistration(T5ForConditionalGeneration, task="seq2seq"),
    "xlm-prophetnet": ModelRegistration(BartForConditionalGeneration, task="seq2seq"),
    # --- Vision ---
    "beit": ModelRegistration(ViTModel, task="image-classification"),
    "blip": ModelRegistration(BlipVisionModel, task="image-classification"),
    "clip_vision_model": ModelRegistration(CLIPVisionModel, task="image-classification"),
    "cvt": ModelRegistration(ViTModel, task="image-classification"),
    "data2vec-vision": ModelRegistration(ViTModel, task="image-classification"),
    "deit": ModelRegistration(ViTModel, task="image-classification"),
    "depth_anything": ModelRegistration(
        DepthAnythingForDepthEstimation, task="image-classification"
    ),
    "dinov2": ModelRegistration(ViTModel, task="image-classification"),
    "dinov2_with_registers": ModelRegistration(ViTModel, task="image-classification"),
    "dinov3_vit": ModelRegistration(ViTModel, task="image-classification"),
    "hiera": ModelRegistration(ViTModel, task="image-classification"),
    "ijepa": ModelRegistration(ViTModel, task="image-classification"),
    "mobilevit": ModelRegistration(ViTModel, task="image-classification"),
    "mobilevitv2": ModelRegistration(ViTModel, task="image-classification"),
    "pvt": ModelRegistration(ViTModel, task="image-classification"),
    "pvt_v2": ModelRegistration(ViTModel, task="image-classification"),
    "sam2": ModelRegistration(Sam2VisionModel, task="image-classification"),
    "segformer": ModelRegistration(
        SegformerForSemanticSegmentation, task="image-classification"
    ),
    "siglip": ModelRegistration(SigLIPVisionModel, task="image-classification"),
    "siglip2": ModelRegistration(SigLIPVisionModel, task="image-classification"),
    "siglip2_vision_model": ModelRegistration(SigLIPVisionModel, task="image-classification"),
    "siglip_vision_model": ModelRegistration(SigLIPVisionModel, task="image-classification"),
    "swin": ModelRegistration(ViTModel, task="image-classification"),
    "swin2sr": ModelRegistration(ViTModel, task="image-classification"),
    "swinv2": ModelRegistration(ViTModel, task="image-classification"),
    "vit": ModelRegistration(ViTModel, task="image-classification"),
    "vit_hybrid": ModelRegistration(ViTModel, task="image-classification"),
    "vit_mae": ModelRegistration(ViTModel, task="image-classification"),
    "vit_msn": ModelRegistration(ViTModel, task="image-classification"),
    "yolos": ModelRegistration(YolosForObjectDetection, task="object-detection"),
    # --- Audio ---
    "data2vec-audio": ModelRegistration(Wav2Vec2Model, task="audio-feature-extraction"),
    "hubert": ModelRegistration(Wav2Vec2Model, task="audio-feature-extraction"),
    "mctct": ModelRegistration(Wav2Vec2Model, task="audio-feature-extraction"),
    "musicgen": ModelRegistration(Wav2Vec2Model, task="audio-feature-extraction"),
    "seamless_m4t": ModelRegistration(Wav2Vec2Model, task="audio-feature-extraction"),
    "seamless_m4t_v2": ModelRegistration(Wav2Vec2Model, task="audio-feature-extraction"),
    "sew": ModelRegistration(Wav2Vec2Model, task="audio-feature-extraction"),
    "sew-d": ModelRegistration(Wav2Vec2Model, task="audio-feature-extraction"),
    "speecht5": ModelRegistration(Wav2Vec2Model, task="audio-feature-extraction"),
    "unispeech": ModelRegistration(Wav2Vec2Model, task="audio-feature-extraction"),
    "unispeech-sat": ModelRegistration(Wav2Vec2Model, task="audio-feature-extraction"),
    "voxtral_encoder": ModelRegistration(Wav2Vec2Model, task="audio-feature-extraction"),
    "wav2vec2": ModelRegistration(Wav2Vec2Model, task="audio-feature-extraction"),
    "wav2vec2-bert": ModelRegistration(Wav2Vec2Model, task="audio-feature-extraction"),
    "wav2vec2-conformer": ModelRegistration(Wav2Vec2Model, task="audio-feature-extraction"),
    "wavlm": ModelRegistration(Wav2Vec2Model, task="audio-feature-extraction"),
}


def _create_default_registry() -> ModelRegistry:
    """Create the default registry with all built-in architectures."""
    reg = ModelRegistry()
    for arch, entry in _REGISTRATIONS.items():
        reg.register(
            arch, entry.module_class, task=entry.task, config_class=entry.config_class
        )
    # Attach test_model_id, family, and variant metadata to registrations.
    _apply_test_metadata(reg)
    return reg


# fmt: off
# -- Test model IDs for L2 architecture validation --
# Each maps a registered model_type to a public HuggingFace model.
# Only the config.json is downloaded (no weights).
_TEST_MODEL_IDS: dict[str, str] = {
    # --- CausalLM (Llama-compatible) ---
    "llama": "meta-llama/Llama-3.2-1B",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "qwen2": "Qwen/Qwen2.5-0.5B",
    "cohere": "CohereForAI/c4ai-command-r7b-12-2024",
    "cohere2": "CohereForAI/c4ai-command-r7b-12-2024",
    "exaone": "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
    "glm": "THUDM/glm-4-9b-chat-hf",
    "glm4": "THUDM/glm-4-9b-chat-hf",
    "gpt_neox": "EleutherAI/pythia-70m",
    "gptj": "EleutherAI/gpt-j-6b",
    "stablelm": "stabilityai/stablelm-2-1_6b",
    "starcoder2": "bigcode/starcoder2-3b",
    "yi": "01-ai/Yi-6B",
    "code_llama": "meta-llama/CodeLlama-7b-hf",
    "llama4_text": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "ministral": "mistralai/Ministral-8B-Instruct-2410",
    "baichuan": "baichuan-inc/Baichuan2-7B-Chat",
    "apertus": "swiss-ai/Apertus-8B-Instruct-2509",
    "arcee": "arcee-ai/AFM-4.5B-Base",
    "diffllama": "kajuma/DiffLlama-0.3B-handcut",
    "doge": "SmallDoge/Doge-20M",
    "dots1": "rednote-hilab/dots.llm1.inst",
    "exaone4": "LGAI-EXAONE/EXAONE-4.0-1.2B",
    "helium": "kyutai/helium-1-preview-2b",
    "minicpm": "optimum-intel-internal-testing/tiny-random-minicpm",
    "minicpm3": "openbmb/MiniCPM3-4B",
    "ministral3": "Aratako/Ministral-3-3B-Instruct-2512-BF16-TextOnly",
    "nanochat": "nanochat-students/nanochat-d20",
    "olmo3": "allenai/Olmo-3-7B-Instruct",
    "openelm": "apple/OpenELM-270M",
    "youtu": "tencent/Youtu-LLM-2B-Base",
    "zamba": "Zyphra/Zamba-7B-v1",
    "zamba2": "Zyphra/Zamba2-1.2B",
    "codegen2": "Salesforce/codegen2-1B",
    "command_r": "CohereForAI/c4ai-command-r-v01",
    "csm": "sesame/csm-1b",
    "evolla": "westlake-repl/Evolla-10B-hf",
    "nemotron_h": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    "open-llama": "openlm-research/open_llama_3b",
    "persimmon": "adept/persimmon-8b-base",
    "shieldgemma2": "google/shieldgemma-2b",
    "solar_open": "upstage/solar-pro-preview-instruct",

    # --- CausalLM (architecture-specific) ---
    "falcon": "tiiuae/falcon-7b",
    "bloom": "bigscience/bloom-560m",
    "gemma": "google/gemma-2b",
    "gemma2": "google/gemma-2-2b",
    "gemma3": "google/gemma-3-4b-it",
    "gemma3_text": "google/gemma-3-1b-pt",
    "gemma3n": "google/gemma-3n-E2B-pt",
    "gemma3n_text": "google/gemma-3n-E2B-pt",
    "gemma4_text": "google/gemma-4-E2B-it",
    "granite": "ibm-granite/granite-3.3-2b-instruct",
    "internlm2": "internlm/internlm2_5-7b-chat",
    "nemotron": "nvidia/Nemotron-Mini-4B-Instruct",
    "olmo": "allenai/OLMo-1B-hf",
    "olmo2": "allenai/OLMo-2-1124-7B",
    "phi": "microsoft/phi-1_5",
    "phi3": "microsoft/Phi-3.5-mini-instruct",
    "phi3small": "microsoft/Phi-3-small-8k-instruct",
    "qwen": "Qwen/Qwen-1_8B-Chat",
    "qwen3": "Qwen/Qwen3-0.6B",
    "qwen3_5_text": "Qwen/Qwen3.5-2B",
    "smollm3": "HuggingFaceTB/SmolLM3-3B",
    "gpt2": "openai-community/gpt2",
    "opt": "facebook/opt-125m",
    "mpt": "mosaicml/mpt-7b",
    "biogpt": "microsoft/biogpt",
    "chatglm": "zai-org/chatglm2-6b",
    "codegen": "Salesforce/codegen-350M-mono",
    "ctrl": "Salesforce/ctrl",
    "ernie4_5": "baidu/ERNIE-4.5-0.3B-PT",
    "falcon_h1": "tiiuae/Falcon-H1-0.5B-Base",
    "gpt-sw3": "AI-Sweden-Models/gpt-sw3-356m",
    "gpt_bigcode": "bigcode/gpt_bigcode-santacoder",
    "gpt_neo": "EleutherAI/gpt-neo-125m",
    "gpt_neox_japanese": "abeja/gpt-neox-japanese-2.7b",
    "gpt_oss": "openai/gpt-oss-20b",
    "hunyuan_v1_dense": "optimum-intel-internal-testing/tiny-random-hunyuan-v1-dense",
    "hunyuan_v1_moe": "tencent/Hunyuan-A13B-Instruct",
    "imagegpt": "openai/imagegpt-small",
    "openai-gpt": "openai-community/openai-gpt",
    "xglm": "facebook/xglm-564M",
    "xlm": "FacebookAI/xlm-mlm-en-2048",

    # --- Mixture of Experts ---
    "mixtral": "mistralai/Mixtral-8x7B-v0.1",
    "phimoe": "microsoft/Phi-tiny-MoE-instruct",
    "qwen2_moe": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    "qwen3_moe": "Qwen/Qwen3-30B-A3B",
    "qwen3_5_moe": "Qwen/Qwen3.5-MoE-A3B-128K",
    "qwen3_next": "Qwen/Qwen3-235B-A22B",
    "granitemoe": "ibm-granite/granite-3.0-1b-a400m-instruct",
    "olmoe": "allenai/OLMoE-1B-7B-0924",
    "dbrx": "databricks/dbrx-instruct",
    "arctic": "Snowflake/snowflake-arctic-instruct",
    "jetmoe": "jetmoe/jetmoe-8b",
    "longcat_flash": "yujiepan/longcat-flash-tiny-random",
    "minimax": "MiniMaxAI/MiniMax-Text-01",
    "ernie4_5_moe": "baidu/ERNIE-4.5-21B-A3B-PT",
    "flex_olmo": "allenai/Flex-reddit-2x7B-1T",
    "glm4_moe": "zai-org/GLM-4.5-Air",
    "granitemoehybrid": "ibm-granite/granite-4.0-tiny-preview",
    "granitemoeshared": "ibm-research/moe-7b-1b-active-shared-experts",
    "qwen3_omni_moe": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "qwen3_vl_moe": "Qwen/Qwen3-VL-30B-A3B-Instruct",

    # --- DeepSeek (MLA + MoE) ---
    "deepseek_v2": "deepseek-ai/DeepSeek-V2-Lite",
    "deepseek_v2_moe": "deepseek-ai/DeepSeek-V2-Lite",
    "deepseek_v3": "deepseek-ai/DeepSeek-V3",

    # --- SSM (Mamba) ---
    "mamba": "state-spaces/mamba-130m-hf",
    "mamba2": "state-spaces/mamba2-130m",
    "falcon_mamba": "tiiuae/falcon-mamba-7b",

    # --- Hybrid SSM+Attention ---
    "jamba": "ai21labs/Jamba-v0.1",
    "bamba": "ibm-fms/Bamba-9B",

    # --- Multimodal ---
    "qwen2_vl": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2_vl_text": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2_5_vl": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2_5_vl_text": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen3_vl": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3_vl_text": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3_5": "Qwen/Qwen3.5-2B",
    "llava": "llava-hf/llava-1.5-7b-hf",
    "llava_next": "llava-hf/llava-v1.6-mistral-7b-hf",
    "mllama": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "gemma4": "google/gemma-4-E2B-it",
    "internvl2": "OpenGVLab/InternVL2-1B",
    "phi4mm": "microsoft/Phi-4-multimodal-instruct",
    "phi4_multimodal": "microsoft/Phi-4-multimodal-instruct",
    "blip-2": "Salesforce/blip2-opt-2.7b",
    "florence2": "microsoft/Florence-2-base",
    "idefics2": "HuggingFaceM4/idefics2-8b",
    "idefics3": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "instructblip": "Salesforce/instructblip-flan-t5-xl",
    "llava_onevision": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    "molmo": "allenai/MolmoE-1B-0924",
    "mistral3": "mistralai/Ministral-3-3B-Instruct-2512",
    "aya_vision": "CohereForAI/aya-vision-8b",
    "chameleon": "facebook/chameleon-7b",
    "cohere2_vision": "CohereForAI/c4ai-command-r7b-12-2024",
    "deepseek_vl": "deepseek-ai/deepseek-vl2-tiny",
    "deepseek_vl_hybrid": "deepseek-ai/deepseek-vl2-tiny",
    "deepseek_vl_v2": "deepseek-ai/deepseek-vl2-tiny",
    "fuyu": "adept/fuyu-8b",
    "glm4v": "THUDM/glm-4v-9b",
    "glm4v_moe": "THUDM/glm-4v-9b",
    "glm4v_moe_text": "THUDM/glm-4v-9b",
    "glm4v_text": "THUDM/glm-4v-9b",
    "got_ocr2": "stepfun-ai/GOT-OCR2_0",
    "instructblipvideo": "Salesforce/instructblip-flan-t5-xl",
    "internvl": "OpenGVLab/InternVL2-1B",
    "internvl_chat": "OpenGVLab/InternVL-Chat-V1-5",
    "janus": "deepseek-ai/Janus-Pro-1B",
    "llava_next_video": "llava-hf/LLaVA-NeXT-Video-7B-hf",
    "ovis2": "AIDC-AI/Ovis2-1B",
    "paligemma": "google/paligemma-3b-pt-224",
    "pixtral": "mistralai/Pixtral-12B-2409",
    "qwen3_5_vl": "Qwen/Qwen3.5-2B",
    "qwen3_5_vl_text": "Qwen/Qwen3.5-2B",
    "qwen3_vl_single": "Qwen/Qwen3-VL-2B-Instruct",
    "sam2": "facebook/sam2-hiera-base-plus",
    "smolvlm": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "video_llava": "LanguageBind/Video-LLaVA-7B-hf",
    "vipllava": "llava-hf/vip-llava-7b-hf",

    # --- Speech ---
    "whisper": "openai/whisper-tiny",
    "qwen3_asr": "Qwen/Qwen3-ASR-0.6B",
    "fun_asr": "justinchuby/Fun-ASR-Nano-2512",
    "speecht5": "microsoft/speecht5_asr",
    "sew": "asapp/sew-tiny-100k",
    "sew-d": "asapp/sew-d-tiny-100k",
    "unispeech": "optimum-intel-internal-testing/tiny-random-unispeech",
    "unispeech-sat": "optimum-intel-internal-testing/tiny-random-UnispeechSatModel",
    "wav2vec2-bert": "facebook/w2v-bert-2.0",
    "wavlm": "microsoft/wavlm-base-plus",
    "data2vec-audio": "facebook/data2vec-audio-base-960h",
    "musicgen": "facebook/musicgen-small",
    "seamless_m4t": "facebook/hf-seamless-m4t-medium",
    "seamless_m4t_v2": "facebook/seamless-m4t-v2-large",
    "mctct": "speechbrain/m-ctc-t-large",
    "qwen3_forced_aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
    "qwen3_tts": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "qwen3_tts_tokenizer_12hz": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "voxtral_encoder": "mistralai/Ministral-3-3B-Instruct-2512",

    # --- Encoder-only ---
    "bert": "google-bert/bert-base-uncased",
    "distilbert": "distilbert/distilbert-base-uncased",
    "roberta": "FacebookAI/roberta-base",
    "albert": "albert/albert-base-v2",
    "electra": "google/electra-small-generator",
    "deberta": "microsoft/deberta-base",
    "deberta-v2": "microsoft/deberta-v3-base",
    "xlm-roberta": "FacebookAI/xlm-roberta-base",
    "modernbert": "answerdotai/ModernBERT-base",
    "modernbert-decoder": "answerdotai/ModernBERT-base",
    "megatron-bert": "nvidia/megatron-bert-uncased-345m",
    "qdqbert": "google-bert/bert-base-uncased",
    "clip_text_model": "openai/clip-vit-base-patch32",
    "bros": "naver-clova-ocr/bros-base-uncased",
    "camembert": "almanach/camembert-base",
    "data2vec-text": "facebook/data2vec-text-base",
    "ernie": "nghuyong/ernie-3.0-base-zh",
    "ernie_m": "Xenova/tiny-random-ErnieMModel",
    "esm": "facebook/esm2_t6_8M_UR50D",
    "flaubert": "flaubert/flaubert_base_cased",
    "ibert": "optimum-intel-internal-testing/tiny-random-ibert",
    "layoutlm": "microsoft/layoutlm-base-uncased",
    "layoutlmv2": "microsoft/layoutlmv2-base-uncased",
    "lilt": "SCUT-DLVCLab/lilt-roberta-en-base",
    "markuplm": "microsoft/markuplm-base",
    "mega": "mnaylor/mega-base-wikitext",
    "mobilebert": "google/mobilebert-uncased",
    "mpnet": "microsoft/mpnet-base",
    "mra": "uw-madison/mra-base-512-4",
    "nezha": "sijunhe/nezha-cn-base",
    "nystromformer": "optimum-intel-internal-testing/tiny-random-NystromformerModel",
    "rembert": "google/rembert",
    "roberta-prelayernorm": "andreasmadsen/efficient_mlm_m0.40",
    "roc_bert": "weiweishi/roc-bert-base-zh",
    "roformer": "junnyu/roformer_chinese_small",
    "splinter": "tau/splinter-base",
    "squeezebert": "optimum-intel-internal-testing/tiny-random-squeezebert",
    "xlm-roberta-xl": "facebook/xlm-roberta-xl",
    "xlnet": "xlnet/xlnet-base-cased",
    "xmod": "facebook/xmod-base",
    "yoso": "uw-madison/yoso-4096",

    # --- Encoder-decoder ---
    "bart": "facebook/bart-base",
    "t5": "google-t5/t5-small",
    "mt5": "google/mt5-small",
    "marian": "Helsinki-NLP/opus-mt-en-de",
    "mbart": "facebook/mbart-large-cc25",
    "pegasus": "google/pegasus-xsum",
    "trocr": "microsoft/trocr-small-handwritten",
    "bigbird_pegasus": "google/bigbird-pegasus-large-bigpatent",
    "blenderbot": "facebook/blenderbot-400M-distill",
    "blenderbot-small": "facebook/blenderbot_small-90M",
    "fsmt": "stas/tiny-wmt19-en-de",
    "led": "allenai/led-base-16384",
    "longt5": "google/long-t5-tglobal-base",
    "m2m_100": "facebook/m2m100_418M",
    "mvp": "RUCAIBox/mvp",
    "pegasus_x": "google/pegasus-x-base",
    "plbart": "uclanlp/plbart-base",
    "prophetnet": "microsoft/prophetnet-large-uncased",
    "switch_transformers": "google/switch-base-8",
    "umt5": "IMISLab/GreekT5-umt5-small-greeksum",
    "xlm-prophetnet": "microsoft/xprophetnet-large-wiki100-cased",
    "nllb-moe": "facebook/nllb-moe-54b",
    "nllb_moe": "facebook/nllb-moe-54b",

    # --- Vision ---
    "vit": "google/vit-base-patch16-224",
    "dinov2": "facebook/dinov2-small",
    "beit": "microsoft/beit-base-patch16-224",
    "clip_vision_model": "openai/clip-vit-base-patch32",
    "swin": "microsoft/swin-tiny-patch4-window7-224",
    "deit": "facebook/deit-small-patch16-224",
    "blip": "Salesforce/blip-image-captioning-base",
    "depth_anything": "LiheYoung/depth-anything-small-hf",
    "yolos": "hustvl/yolos-tiny",
    "segformer": "nvidia/segformer-b0-finetuned-ade-512-512",
    "cvt": "microsoft/cvt-13",
    "data2vec-vision": "facebook/data2vec-vision-base-ft1k",
    "dinov2_with_registers": "facebook/dinov2-with-registers-base",
    "hiera": "facebook/hiera-tiny-224-mae-hf",
    "layoutlmv3": "microsoft/layoutlmv3-base",
    "mobilevit": "apple/mobilevit-small",
    "mobilevitv2": "apple/mobilevitv2-1.0-imagenet1k-256",
    "pvt": "Zetatech/pvt-tiny-224",
    "pvt_v2": "OpenGVLab/pvt_v2_b0",
    "siglip": "google/siglip-base-patch16-224",
    "siglip2": "google/siglip2-base-patch16-224",
    "siglip_vision_model": "google/siglip-base-patch16-224",
    "siglip2_vision_model": "google/siglip2-base-patch16-224",
    "swin2sr": "caidas/swin2SR-classical-sr-x2-64",
    "swinv2": "microsoft/swinv2-tiny-patch4-window16-256",
    "vit_mae": "facebook/vit-mae-base",
    "vit_msn": "facebook/vit-msn-small",
    "dinov3_vit": "facebook/dinov2-small",
    "ijepa": "facebook/ijepa_vith14_1k",
    "vit_hybrid": "google/vit-hybrid-base-bit-384",

    # --- Audio ---
    "wav2vec2": "facebook/wav2vec2-base",
    "hubert": "facebook/hubert-base-ls960",
    "wav2vec2-conformer": "facebook/wav2vec2-conformer-rope-large-960h-ft",
}
# fmt: on

# -- Family overrides for dashboard grouping --
# Models that share an architecture family but have different model_type prefixes.
_FAMILY_OVERRIDES: dict[str, str] = {
    "phi": "phi",
    "phi3": "phi",
    "phi3small": "phi",
    "phimoe": "phi",
    "phi4mm": "phi",
    "phi4_multimodal": "phi",
    "gemma": "gemma",
    "gemma2": "gemma",
    "shieldgemma2": "gemma",
    "gemma3": "gemma",
    "gemma3_text": "gemma",
    "gemma3n": "gemma",
    "gemma3n_text": "gemma",
    "internlm2": "internlm",
    "internvl_chat": "internlm",
    "internvl2": "internlm",
    "internvl": "internlm",
    "qwen": "qwen",
    "qwen2": "qwen",
    "qwen2_moe": "qwen",
    "qwen3": "qwen",
    "qwen3_moe": "qwen",
    "qwen3_5_text": "qwen",
    "qwen3_5_moe": "qwen",
    "qwen3_next": "qwen",
    "qwen2_vl": "qwen",
    "qwen2_vl_text": "qwen",
    "qwen2_5_vl": "qwen",
    "qwen2_5_vl_text": "qwen",
    "qwen3_vl": "qwen",
    "qwen3_vl_text": "qwen",
    "qwen3_vl_single": "qwen",
    "qwen3_vl_moe": "qwen",
    "qwen3_5": "qwen",
    "qwen3_5_vl": "qwen",
    "qwen3_omni_moe": "qwen",
    "qwen3_asr": "qwen",
    "qwen3_forced_aligner": "qwen",
    "fun_asr": "qwen",
    "qwen3_tts": "qwen",
    "qwen3_tts_tokenizer_12hz": "qwen",
    "deepseek_v2": "deepseek",
    "deepseek_v2_moe": "deepseek",
    "deepseek_v3": "deepseek",
    "deepseek_vl_v2": "deepseek",
    "olmo": "olmo",
    "olmo2": "olmo",
    "olmo3": "olmo",
    "olmoe": "olmo",
    "llama": "llama",
    "code_llama": "llama",
    "llama4_text": "llama",
    "mllama": "llama",
    "mistral": "mistral",
    "mistral3": "mistral",
    "ministral": "mistral",
    "ministral3": "mistral",
    "mixtral": "mistral",
    "pixtral": "mistral",
    "falcon": "falcon",
    "falcon_h1": "falcon",
    "falcon_mamba": "falcon",
    "mamba": "mamba",
    "mamba2": "mamba",
    "bloom": "bloom",
    "gpt2": "gpt2",
    "gpt_neo": "gpt2",
    "gpt_bigcode": "gpt2",
    "gpt_neox": "gpt_neox",
    "gpt_neox_japanese": "gpt_neox",
    "bart": "bart",
    "mbart": "bart",
    "t5": "t5",
    "mt5": "t5",
    "longt5": "t5",
    "umt5": "t5",
    "switch_transformers": "t5",
    "bert": "bert",
    "albert": "bert",
    "roberta": "bert",
    "xlm-roberta": "bert",
    "xlm-roberta-xl": "bert",
    "distilbert": "bert",
    "deberta": "deberta",
    "deberta-v2": "deberta",
    "wav2vec2": "wav2vec2",
    "wav2vec2-bert": "wav2vec2",
    "wav2vec2-conformer": "wav2vec2",
    "hubert": "wav2vec2",
    "wavlm": "wav2vec2",
    "vit": "vit",
    "vit_hybrid": "vit",
    "vit_mae": "vit",
    "vit_msn": "vit",
    "deit": "vit",
    "beit": "vit",
    "dinov2": "vit",
    "dinov2_with_registers": "vit",
    "swin": "swin",
    "swin2sr": "swin",
    "swinv2": "swin",
    "clip_text_model": "clip",
    "clip_vision_model": "clip",
    "siglip": "clip",
    "siglip2": "clip",
    "siglip_vision_model": "clip",
    "siglip2_vision_model": "clip",
}

# -- Variant labels for code-path identification --
_VARIANT_LABELS: dict[str, str] = {
    "deepseek_v2": "mla",
    "deepseek_v2_moe": "mla+moe",
    "deepseek_v3": "mla+moe",
    "phi3small": "blocksparse",
    "falcon_h1": "hybrid-ssm",
    "mamba": "ssm",
    "mamba2": "ssm",
    "falcon_mamba": "ssm",
    "jamba": "hybrid-ssm+attn",
    "bamba": "hybrid-mamba2+attn",
    "qwen3_next": "moe+linear-attn",
}


def _apply_test_metadata(reg: ModelRegistry) -> None:
    """Apply test_model_id, family, and variant metadata to registrations.

    Called at the end of ``_create_default_registry()`` to attach test
    metadata without disturbing the existing registration patterns.
    """
    for arch, model_id in _TEST_MODEL_IDS.items():
        if arch in reg:
            old = reg._map[arch]
            reg._map[arch] = dataclasses.replace(old, test_model_id=model_id)
    for arch, family in _FAMILY_OVERRIDES.items():
        if arch in reg:
            old = reg._map[arch]
            reg._map[arch] = dataclasses.replace(old, family=family)
    for arch, variant in _VARIANT_LABELS.items():
        if arch in reg:
            old = reg._map[arch]
            reg._map[arch] = dataclasses.replace(old, variant=variant)


#: The default model registry with all built-in architectures.
registry: ModelRegistry = _create_default_registry()

# Backward-compatible alias — exposes internal dict directly.
# Deprecated: use registry.get() / registry.register() instead.
# This will be removed in a future version.
MODEL_MAP = registry._map
