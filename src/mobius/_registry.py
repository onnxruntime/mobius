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
    CodeShellConfig,
    Eagle3Config,
    FalconH1Config,
    Gemma3nMultiModalConfig,
    Gemma4AssistantConfig,
    Gemma4Config,
    Jais2Config,
    KimiK3Config,
    KimiLinearConfig,
    Lfm2Config,
    Lfm2MoeConfig,
    Lfm2VlConfig,
    MiniMaxConfig,
    MMSConfig,
    MoonshineConfig,
    MuseGlimmerConfig,
    NemotronParseConfig,
    ParakeetCTCConfig,
    Plamo2Config,
    Qwen4ExpConfig,
    SenseNovaU1Config,
    WhisperConfig,
    XverseConfig,
)
from mobius.models import (
    ApertusCausalLMModel,
    ArceeCausalLMModel,
    ArcticGGUFCausalLMModel,
    BitNetCausalLMModel,
    CausalLMModel,
    ChatGLMCausalLMModel,
    CodeShellCausalLMModel,
    Cosmos3EdgeTextModel,
    Cosmos3EdgeVLModel,
    Cosmos3OmniReasonerModel,
    DbrxGGUFCausalLMModel,
    DeepSeekOCR2CausalLMModel,
    DeepSeekV3CausalLMModel,
    DeepSeekV4CausalLMModel,
    DFlashDraftModel,
    DiffLlamaCausalLMModel,
    DogeCausalLMModel,
    DreamModel,
    Eagle3DraftModel,
    EncDecRNNTModel,
    Ernie45MoECausalLMModel,
    Ernie45MoEGGUFCausalLMModel,
    ErnieCausalLMModel,
    ExaOne4CausalLMModel,
    Gemma2CausalLMModel,
    Gemma3CausalLMModel,
    Gemma3MultiModalModel,
    Gemma4AssistantCausalLMModel,
    Gemma4CausalLMModel,
    Gemma4Model,
    Gemma4UnifiedModel,
    GemmaCausalLMModel,
    Glm4CausalLMModel,
    Glm4MoECausalLMModel,
    GlmCausalLMModel,
    GlmMoeDsaCausalLMModel,
    GlmOcrForConditionalGeneration,
    GPTOSSCausalLMModel,
    GraniteCausalLMModel,
    GraniteMoECausalLMModel,
    HunYuanMoEV1CausalLMModel,
    HunYuanV1DenseCausalLMModel,
    HunYuanVLMoTModel,
    InternLM2CausalLMModel,
    Jais2CausalLMModel,
    KimiK3CausalLMModel,
    KimiLinearCausalLMModel,
    LayerNormCausalLMModel,
    Lfm2CausalLMModel,
    Lfm2MoECausalLMModel,
    Lfm2VlForConditionalGeneration,
    LLaDAModel,
    LLaDAMoEModel,
    Llama4CausalLMModel,
    MageVLForConditionalGeneration,
    MoECausalLMModel,
    MoonshineForConditionalGeneration,
    NanoChatCausalLMModel,
    NemotronCausalLMModel,
    NemotronParseForConditionalGeneration,
    OLMo2CausalLMModel,
    OLMoCausalLMModel,
    ParakeetForCTCModel,
    Phi3CausalLMModel,
    Phi3MoECausalLMModel,
    Phi3SmallCausalLMModel,
    Phi3VModel,
    Phi4MMMultiModalModel,
    Phi4SigLIPModel,
    PhiCausalLMModel,
    PhiMoEGGUFCausalLMModel,
    Plamo2ForCausalLM,
    PLMCausalLMModel,
    Qwen2MoECausalLMModel,
    Qwen2VLCausalLMModel,
    Qwen3CausalLMModel,
    Qwen3NextCausalLMModel,
    Qwen3VL3ModelCausalLMModel,
    Qwen3VLCausalLMModel,
    Qwen3VLTextModel,
    Qwen4ExpCausalLMModel,
    Qwen4ExpForConditionalGeneration,
    Qwen25VLCausalLMModel,
    Qwen25VLTextModel,
    Qwen35CausalLMModel,
    Qwen35MoECausalLMModel,
    Qwen35MoEVL3ModelCausalLMModel,
    Qwen35MtpModel,
    Qwen35VL3ModelCausalLMModel,
    Qwen35VLTextModel,
    QwenCausalLMModel,
    RND1Model,
    SmolLM3CausalLMModel,
    SortformerDiarizationModel,
    WhisperForConditionalGeneration,
    XverseCausalLMModel,
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
from mobius.models.esm import EsmConfig, EsmModel
from mobius.models.falcon import (
    BloomCausalLMModel,
    FalconCausalLMModel,
    MPTCausalLMModel,
)
from mobius.models.falcon_h1 import FalconH1ForCausalLM
from mobius.models.fun_asr import FunASRForConditionalGeneration
from mobius.models.gemma3n import Gemma3nCausalLMModel, Gemma3nMultiModalModel
from mobius.models.gguf_embeddings import GemmaEmbeddingGGUFModel, LlamaEmbedGGUFModel
from mobius.models.gguf_encoders import (
    EuroBertGGUFModel,
    JinaBertV2GGUFModel,
    NeoBertGGUFModel,
    NomicBertGGUFModel,
    NomicBertMoEGGUFModel,
)
from mobius.models.gguf_legacy_decoders import ExactLegacyGGUFCausalLMModel
from mobius.models.glm_asr import GlmAsrForConditionalGeneration
from mobius.models.gpt2 import GPT2CausalLMModel, ScaledEmbeddingGPT2CausalLMModel
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
from mobius.models.minicpm import MiniCPM3CausalLMModel, MiniCPMCausalLMModel
from mobius.models.minicpmv4_6 import MiniCPMV46ForConditionalGeneration
from mobius.models.minimax import MiniMaxCausalLMModel
from mobius.models.mllama import MllamaCausalLMModel
from mobius.models.modernbert import ModernBertDecoderModel, ModernBertModel
from mobius.models.muse_glimmer import (
    MuseGlimmerForConditionalGeneration,
    MuseGlimmerTextCausalLMModel,
)
from mobius.models.nemotron_h import NemotronHCausalLMModel
from mobius.models.opt import OPTCausalLMModel
from mobius.models.persimmon import PersimmonCausalLMModel
from mobius.models.qwen3_asr import Qwen3ASRForConditionalGeneration
from mobius.models.qwen3_tts import Qwen3TTSForConditionalGeneration
from mobius.models.qwen3_tts_tokenizer import Qwen3TTSTokenizerV2Model
from mobius.models.sam2 import Sam2VisionModel
from mobius.models.segformer import SegformerForSemanticSegmentation
from mobius.models.sensenova_u1 import SenseNovaU1Model
from mobius.models.sensevoice_small import SenseVoiceSmallModel
from mobius.models.starcoder2 import StarCoder2CausalLMModel
from mobius.models.t5 import T5EncoderModel, T5ForConditionalGeneration
from mobius.models.talkie import TalkieForCausalLM
from mobius.models.trocr import TrOCRForConditionalGeneration
from mobius.models.vit import ViTModel
from mobius.models.wav2vec2 import Wav2Vec2Model
from mobius.models.wav2vec2_ctc import Wav2Vec2ForCTCModel
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
        test_revision: Optional immutable HuggingFace revision for L2 validation.
    """

    module_class: type[nn.Module]
    task: str | None = None
    config_class: type[BaseModelConfig] | None = None
    test_model_id: str | None = None
    family: str | None = None
    variant: str | None = None
    test_revision: str | None = None


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
        test_revision: str | None = None,
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
            test_revision: Optional immutable HuggingFace revision for L2 validation.
        """
        self._map[architecture] = ModelRegistration(
            module_class,
            task,
            config_class,
            test_model_id,
            family,
            variant,
            test_revision,
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
    "command_r": ModelRegistration(CohereCausalLMModel),
    "jais2": ModelRegistration(Jais2CausalLMModel, config_class=Jais2Config),
    "kclgpt": ModelRegistration(CodeShellCausalLMModel, config_class=CodeShellConfig),
    "csm": ModelRegistration(CausalLMModel),
    "dots1": ModelRegistration(DeepSeekV3CausalLMModel),
    "deepseek": ModelRegistration(DeepSeekV3CausalLMModel),
    "evolla": ModelRegistration(CausalLMModel),
    "exaone": ModelRegistration(CausalLMModel),
    "helium": ModelRegistration(CausalLMModel),
    "llama": ModelRegistration(CausalLMModel),
    "minicpm": ModelRegistration(CausalLMModel),
    "minicpm3": ModelRegistration(CausalLMModel),
    "minicpm_gguf": ModelRegistration(MiniCPMCausalLMModel),
    "minicpm3_gguf": ModelRegistration(MiniCPM3CausalLMModel),
    "ministral": ModelRegistration(CausalLMModel),
    "ministral3": ModelRegistration(CausalLMModel),
    "mistral": ModelRegistration(CausalLMModel),
    "open-llama": ModelRegistration(CausalLMModel),
    "xverse": ModelRegistration(XverseCausalLMModel, config_class=XverseConfig),
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
    "bitnet": ModelRegistration(BitNetCausalLMModel),
    "talkie": ModelRegistration(TalkieForCausalLM),
    "bloom": ModelRegistration(BloomCausalLMModel),
    "orion": ModelRegistration(LayerNormCausalLMModel),
    "chatglm": ModelRegistration(ChatGLMCausalLMModel),
    "codegen": ModelRegistration(CodeGenCausalLMModel),
    "cohere": ModelRegistration(CohereCausalLMModel),
    "cohere2": ModelRegistration(CohereCausalLMModel),
    "cosmos3_edge": ModelRegistration(Cosmos3EdgeVLModel),
    "cosmos3_edge_text": ModelRegistration(Cosmos3EdgeTextModel),
    "cosmos3_omni": ModelRegistration(
        Cosmos3OmniReasonerModel, task="qwen-vl", family="cosmos", variant="reasoner"
    ),
    "diffllama": ModelRegistration(DiffLlamaCausalLMModel),
    "doge": ModelRegistration(DogeCausalLMModel),
    "ernie4_5": ModelRegistration(ErnieCausalLMModel),
    "exaone4": ModelRegistration(ExaOne4CausalLMModel),
    "falcon": ModelRegistration(FalconCausalLMModel),
    "falcon_h1": ModelRegistration(
        FalconH1ForCausalLM,
        task="falcon-h1-text-generation",
        config_class=FalconH1Config,
        test_model_id="tiiuae/Falcon-H1-Tiny-90M-Base",
        family="falcon-h1",
    ),
    "plamo2": ModelRegistration(
        Plamo2ForCausalLM,
        task="plamo2-text-generation",
        config_class=Plamo2Config,
        test_model_id="pfnet/plamo-2-1b",
        family="plamo2",
    ),
    "plm": ModelRegistration(
        PLMCausalLMModel,
        test_model_id="PLM-Team/PLM-1.8B-Instruct",
        test_revision="62d188c7d58843d7013d5b3ffe198db448787860",
        family="plm",
    ),
    "pangu_embedded": ModelRegistration(CausalLMModel, family="pangu-embedded"),
    "gemma": ModelRegistration(GemmaCausalLMModel),
    "gemma2": ModelRegistration(Gemma2CausalLMModel),
    "gemma3": ModelRegistration(Gemma3MultiModalModel, task="gemma3-vision-language"),
    "gemma3_text": ModelRegistration(Gemma3CausalLMModel),
    "gemma3n_text": ModelRegistration(Gemma3nCausalLMModel),
    "gemma4_text": ModelRegistration(Gemma4CausalLMModel, config_class=Gemma4Config),
    "gemma4_unified_text": ModelRegistration(Gemma4CausalLMModel, config_class=Gemma4Config),
    # Gemma4-Assistant ships ``model_type="gemma4_assistant"`` plus
    # ``architectures=["Gemma4AssistantForCausalLM"]``.  Both keys point
    # at the assistant drafter so build() can dispatch via either.  The
    # ``gemma4_unified_assistant`` family is structurally identical (same
    # 4-layer Gemma4 decoder, KV-shared with target, optional ordered-
    # embeddings head) but ships under a different ``model_type``;
    # register all four keys at the same class.
    "gemma4_assistant": ModelRegistration(
        Gemma4AssistantCausalLMModel,
        task="gemma4-assistant",
        config_class=Gemma4AssistantConfig,
        family="gemma4_assistant",
    ),
    "Gemma4AssistantForCausalLM": ModelRegistration(
        Gemma4AssistantCausalLMModel,
        task="gemma4-assistant",
        config_class=Gemma4AssistantConfig,
        family="gemma4_assistant",
    ),
    "gemma4_unified_assistant": ModelRegistration(
        Gemma4AssistantCausalLMModel,
        task="gemma4-assistant",
        config_class=Gemma4AssistantConfig,
        family="gemma4_assistant",
        variant="unified",
    ),
    "Gemma4UnifiedAssistantForCausalLM": ModelRegistration(
        Gemma4AssistantCausalLMModel,
        task="gemma4-assistant",
        config_class=Gemma4AssistantConfig,
        family="gemma4_assistant",
        variant="unified",
    ),
    "glm": ModelRegistration(GlmCausalLMModel),
    "glm4": ModelRegistration(Glm4CausalLMModel),
    "gpt_neox": ModelRegistration(GPTNeoXCausalLMModel),
    "gguf_legacy": ModelRegistration(ExactLegacyGGUFCausalLMModel),
    "gpt_neox_japanese": ModelRegistration(GPTNeoXJapaneseCausalLMModel),
    "gpt_oss": ModelRegistration(GPTOSSCausalLMModel),
    "gptj": ModelRegistration(GPTJCausalLMModel),
    "granite": ModelRegistration(GraniteCausalLMModel),
    "hunyuan_v1_dense": ModelRegistration(HunYuanV1DenseCausalLMModel),
    "internlm2": ModelRegistration(InternLM2CausalLMModel),
    "llama4_text": ModelRegistration(Llama4CausalLMModel),
    "lfm2": ModelRegistration(Lfm2CausalLMModel, config_class=Lfm2Config),
    "lfm2_moe": ModelRegistration(Lfm2MoECausalLMModel, config_class=Lfm2MoeConfig),
    "dream": ModelRegistration(DreamModel, task="masked-diffusion"),
    "Dream": ModelRegistration(DreamModel, task="masked-diffusion"),
    "llada": ModelRegistration(LLaDAModel, task="masked-diffusion"),
    "llada_moe": ModelRegistration(LLaDAMoEModel, task="masked-diffusion"),
    "LLaDAMoEModel": ModelRegistration(LLaDAMoEModel, task="masked-diffusion"),
    "rnd1": ModelRegistration(RND1Model, task="masked-diffusion"),
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
    # DFlash drafters share ``model_type=qwen3`` with the base Qwen3 family;
    # build() routes to DFlashDraftModel by the ``architectures`` field in the
    # checkpoint config.  The registry entry below allows direct lookup by
    # architecture name and exposes the task.
    "DFlashDraftModel": ModelRegistration(
        DFlashDraftModel, task="dflash-draft", family="dflash", variant="qwen3"
    ),
    # Qwen3.6 MTP self-speculative head.  Like DFlash, it is an auxiliary
    # drafter that is NOT auto-routed by ``model_type`` (the main checkpoint
    # maps to the qwen3_5 VL / text model); it is looked up directly by
    # architecture name and exposes the ``qwen35-mtp`` task.
    "Qwen35MtpModel": ModelRegistration(
        Qwen35MtpModel, task="qwen35-mtp", family="qwen", variant="mtp"
    ),
    "Eagle3LlamaForCausalLM": ModelRegistration(
        Eagle3DraftModel,
        task="eagle3-draft",
        config_class=Eagle3Config,
        family="eagle3",
        variant="qwen3",
    ),
    # AngelSlim ships two equivalent EAGLE-3 architecture strings: Qwen3-4B uses
    # "Eagle3LlamaForCausalLM", Qwen3-8B uses "LlamaForCausalLMEagle3".
    "LlamaForCausalLMEagle3": ModelRegistration(
        Eagle3DraftModel,
        task="eagle3-draft",
        config_class=Eagle3Config,
        family="eagle3",
        variant="qwen3",
    ),
    # speculators-format EAGLE-3 checkpoints (RedHat): the Qwen3 checkpoint
    # declares "Eagle3Speculator", the Gemma4 one "Eagle3DraftModel". Both nest
    # the arch config under transformer_layer_config and set norm_before_residual.
    "Eagle3Speculator": ModelRegistration(
        Eagle3DraftModel,
        task="eagle3-draft",
        config_class=Eagle3Config,
        family="eagle3",
        variant="speculators",
    ),
    "Eagle3DraftModel": ModelRegistration(
        Eagle3DraftModel,
        task="eagle3-draft",
        config_class=Eagle3Config,
        family="eagle3",
        variant="speculators",
    ),
    "shieldgemma2": ModelRegistration(Gemma2CausalLMModel),
    "smollm3": ModelRegistration(SmolLM3CausalLMModel),
    "stablelm": ModelRegistration(LayerNormCausalLMModel),
    "starcoder2": ModelRegistration(StarCoder2CausalLMModel),
    # --- Mixture of Experts ---
    "arctic": ModelRegistration(MoECausalLMModel),
    "arctic_gguf": ModelRegistration(ArcticGGUFCausalLMModel),
    "dbrx": ModelRegistration(MoECausalLMModel),
    "dbrx_gguf": ModelRegistration(DbrxGGUFCausalLMModel),
    "ernie4_5_moe": ModelRegistration(Ernie45MoECausalLMModel),
    "ernie4_5_moe_gguf": ModelRegistration(Ernie45MoEGGUFCausalLMModel),
    "bailing_moe": ModelRegistration(Ernie45MoECausalLMModel),
    "flex_olmo": ModelRegistration(MoECausalLMModel),
    "glm4_moe": ModelRegistration(Glm4MoECausalLMModel),
    "granitemoe": ModelRegistration(GraniteMoECausalLMModel),
    "granitemoehybrid": ModelRegistration(GraniteMoeHybridCausalLMModel),
    "granitemoeshared": ModelRegistration(GraniteMoECausalLMModel),
    "hunyuan_v1_moe": ModelRegistration(HunYuanMoEV1CausalLMModel),
    "jetmoe": ModelRegistration(JetMoeCausalLMModel),
    "kimi_linear": ModelRegistration(
        KimiLinearCausalLMModel,
        task="kimi-linear-text-generation",
        config_class=KimiLinearConfig,
    ),
    "kimi_k3": ModelRegistration(
        KimiK3CausalLMModel,
        task="kimi-k3-text-generation",
        config_class=KimiK3Config,
    ),
    "minimax": ModelRegistration(MiniMaxCausalLMModel, config_class=MiniMaxConfig),
    "MiniMaxText01": ModelRegistration(MiniMaxCausalLMModel, config_class=MiniMaxConfig),
    "minimax_text_01": ModelRegistration(MiniMaxCausalLMModel, config_class=MiniMaxConfig),
    "mixtral": ModelRegistration(MoECausalLMModel),
    "olmoe": ModelRegistration(MoECausalLMModel),
    "phimoe": ModelRegistration(Phi3MoECausalLMModel),
    "phimoe_gguf": ModelRegistration(PhiMoEGGUFCausalLMModel),
    "qwen2_moe": ModelRegistration(Qwen2MoECausalLMModel),
    "qwen3_5_moe": ModelRegistration(Qwen35MoECausalLMModel),
    "qwen3_moe": ModelRegistration(MoECausalLMModel),
    "qwen3_next": ModelRegistration(Qwen3NextCausalLMModel),
    "qwen4_exp": ModelRegistration(
        Qwen4ExpForConditionalGeneration,
        task="qwen4-exp-vision-language",
        config_class=Qwen4ExpConfig,
        test_model_id="Qwen/Qwen3.8-Flash-Next",
        family="qwen",
        variant="multimodal+moe+gdn+qsa+ple",
        test_revision="f5d08274bafd880402bd16f5e3e6c514136ec06c",
    ),
    "qwen4_exp_text": ModelRegistration(
        Qwen4ExpCausalLMModel,
        task="qwen4-exp-text-generation",
        config_class=Qwen4ExpConfig,
        test_model_id="unsloth/Qwen3.8-Flash-Next-FP8",
        test_revision="41cc25fe32cc20053a59c89716196897580cddf6",
        family="qwen",
        variant="moe+gdn+qsa+ple",
    ),
    "Qwen4ExpForConditionalGeneration": ModelRegistration(
        Qwen4ExpForConditionalGeneration,
        task="qwen4-exp-vision-language",
        config_class=Qwen4ExpConfig,
        test_model_id="Qwen/Qwen3.8-Flash-Next",
        family="qwen",
        variant="multimodal",
        test_revision="f5d08274bafd880402bd16f5e3e6c514136ec06c",
    ),
    "qwen3_omni_moe": ModelRegistration(MoECausalLMModel),
    "qwen3_vl_moe": ModelRegistration(MoECausalLMModel),
    # --- DeepSeek (MLA + MoE) ---
    "deepseek_v2": ModelRegistration(DeepSeekV3CausalLMModel),
    "deepseek_v2_moe": ModelRegistration(DeepSeekV3CausalLMModel),
    "deepseek_v3": ModelRegistration(DeepSeekV3CausalLMModel),
    "deepseek_v4": ModelRegistration(DeepSeekV4CausalLMModel, task="deepseek-v4"),
    "deepseek_vl_v2": ModelRegistration(DeepSeekOCR2CausalLMModel),
    # --- GLM-5.2 (MLA + DeepSeek Sparse Attention (DSA) + MoE) ---
    "glm_moe_dsa": ModelRegistration(GlmMoeDsaCausalLMModel, task="glm-moe-dsa"),
    # --- SSM (Mamba / Mamba2) ---
    "falcon_mamba": ModelRegistration(MambaCausalLMModel),
    "mamba": ModelRegistration(MambaCausalLMModel),
    "mamba2": ModelRegistration(Mamba2CausalLMModel),
    # --- Hybrid SSM+Attention ---
    "bamba": ModelRegistration(BambaCausalLMModel),
    "jamba": ModelRegistration(JambaCausalLMModel),
    "nemotron_h": ModelRegistration(NemotronHCausalLMModel),
    "nemotron_parse": ModelRegistration(
        NemotronParseForConditionalGeneration,
        task="vision-encoder-decoder",
        config_class=NemotronParseConfig,
    ),
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
    "gemma3n": ModelRegistration(
        Gemma3nMultiModalModel, task="gemma3n", config_class=Gemma3nMultiModalConfig
    ),
    "gemma4": ModelRegistration(Gemma4Model, task="gemma4", config_class=Gemma4Config),
    "gemma4_unified": ModelRegistration(
        Gemma4UnifiedModel, task="gemma4-unified", config_class=Gemma4Config
    ),
    "glm4v": ModelRegistration(LLaVAModel, task="vision-language"),
    "glm4v_moe": ModelRegistration(LLaVAModel, task="vision-language"),
    "glm4v_moe_text": ModelRegistration(Glm4MoECausalLMModel),
    "glm4v_text": ModelRegistration(Glm4CausalLMModel),
    "glm_ocr": ModelRegistration(
        GlmOcrForConditionalGeneration,
        task="glm-ocr",
        test_model_id="zai-org/GLM-OCR",
        test_revision="ca5d8b3e287e52589e37c28385d9655ee4372f9d",
    ),
    "got_ocr2": ModelRegistration(LLaVAModel, task="vision-language"),
    "hunyuan_vl_mot": ModelRegistration(HunYuanVLMoTModel, task="hunyuan-vl-mot"),
    "neo_chat": ModelRegistration(
        SenseNovaU1Model,
        task="sensenova-u1",
        config_class=SenseNovaU1Config,
        family="sensenova",
        variant="mot_unified",
    ),
    "idefics2": ModelRegistration(LLaVAModel, task="vision-language"),
    "idefics3": ModelRegistration(LLaVAModel, task="vision-language"),
    "instructblip": ModelRegistration(LLaVAModel, task="vision-language"),
    "instructblipvideo": ModelRegistration(LLaVAModel, task="vision-language"),
    "internvl": ModelRegistration(InternVL2Model, task="vision-language"),
    "internvl2": ModelRegistration(InternVL2Model, task="vision-language"),
    "internvl_chat": ModelRegistration(InternVL2Model, task="vision-language"),
    "lfm2_vl": ModelRegistration(
        Lfm2VlForConditionalGeneration,
        task="lfm2-vl",
        config_class=Lfm2VlConfig,
    ),
    "mage_vl": ModelRegistration(MageVLForConditionalGeneration, task="mage-vl"),
    "janus": ModelRegistration(LLaVAModel, task="vision-language"),
    "llava": ModelRegistration(LLaVAModel, task="vision-language"),
    "llava_next": ModelRegistration(LLaVAModel, task="vision-language"),
    "llava_next_video": ModelRegistration(LLaVAModel, task="vision-language"),
    "llava_onevision": ModelRegistration(LLaVAModel, task="vision-language"),
    "mistral3": ModelRegistration(LLaVAModel, task="pixtral-vl"),
    "minicpmv4_6": ModelRegistration(
        MiniCPMV46ForConditionalGeneration,
        task="minicpm-vl",
    ),
    "mllama": ModelRegistration(MllamaCausalLMModel, task="mllama-vision-language"),
    "muse_glimmer": ModelRegistration(
        MuseGlimmerForConditionalGeneration,
        task="muse-glimmer-vl",
        config_class=MuseGlimmerConfig,
        test_model_id="meta-models/Muse-Glimmer-30B",
    ),
    "muse_glimmer_text": ModelRegistration(
        MuseGlimmerTextCausalLMModel,
        config_class=MuseGlimmerConfig,
    ),
    "molmo": ModelRegistration(LLaVAModel, task="vision-language"),
    "ovis2": ModelRegistration(LLaVAModel, task="vision-language"),
    "paligemma": ModelRegistration(LLaVAModel, task="vision-language"),
    "phi4_multimodal": ModelRegistration(Phi4MMMultiModalModel, task="phi4mm-multimodal"),
    "phi4mm": ModelRegistration(Phi4MMMultiModalModel, task="phi4mm-multimodal"),
    "phi3_v": ModelRegistration(Phi3VModel, task="vision-language"),
    "phi4-siglip": ModelRegistration(Phi4SigLIPModel, task="vision-language"),
    "pixtral": ModelRegistration(LLaVAModel, task="pixtral-vl"),
    "qwen2_5_vl": ModelRegistration(Qwen25VLCausalLMModel, task="qwen-vl"),
    "qwen2_5_vl_text": ModelRegistration(Qwen25VLTextModel),
    "qwen2_vl": ModelRegistration(Qwen2VLCausalLMModel, task="qwen-vl"),
    "qwen2_vl_text": ModelRegistration(Qwen25VLTextModel),
    "qwen3_5": ModelRegistration(Qwen35VL3ModelCausalLMModel, task="hybrid-qwen-vl"),
    "qwen3_5_moe_vl": ModelRegistration(Qwen35MoEVL3ModelCausalLMModel, task="hybrid-qwen-vl"),
    # Text-only sibling of ``qwen3_5_moe_vl`` (Qwen3.6-35B-A3B). The MoE
    # backbone ``Qwen35MoECausalLMModel`` already strips ``language_model.``
    # and drops ``visual.``/MTP keys, so it consumes the VL checkpoint's text
    # weights directly; ``build(..., text_only=True)`` routes here via
    # ``_TEXT_ONLY_MODEL_TYPE``. It also matches the VL ``text_config``'s own
    # ``model_type=qwen3_5_moe_text`` so that config resolves cleanly.
    "qwen3_5_moe_text": ModelRegistration(Qwen35MoECausalLMModel),
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
    # Fun-ASR uses config.yaml (not config.json). build() auto-detection
    # won't work — use build_from_module() with manual config construction.
    # See examples/fun_asr.py for the full pipeline.
    "fun_asr": ModelRegistration(
        FunASRForConditionalGeneration, task="fun-asr-speech-language"
    ),
    "glmasr": ModelRegistration(GlmAsrForConditionalGeneration, task="glmasr-speech-language"),
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
    "moonshine": ModelRegistration(
        MoonshineForConditionalGeneration,
        task="speech-to-text",
        config_class=MoonshineConfig,
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
    "esm": ModelRegistration(EsmModel, task="feature-extraction", config_class=EsmConfig),
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
    "eurobert_gguf": ModelRegistration(
        EuroBertGGUFModel, task="gguf-encoder-feature-extraction"
    ),
    "jina_bert_v2_gguf": ModelRegistration(
        JinaBertV2GGUFModel, task="gguf-encoder-feature-extraction"
    ),
    "neo_bert_gguf": ModelRegistration(
        NeoBertGGUFModel, task="gguf-encoder-feature-extraction"
    ),
    "nomic_bert_gguf": ModelRegistration(
        NomicBertGGUFModel, task="gguf-encoder-feature-extraction"
    ),
    "nomic_bert_moe_gguf": ModelRegistration(
        NomicBertMoEGGUFModel, task="gguf-encoder-feature-extraction"
    ),
    "gemma_embedding_gguf": ModelRegistration(
        GemmaEmbeddingGGUFModel, task="gguf-embedding-feature-extraction"
    ),
    "llama_embed_gguf": ModelRegistration(
        LlamaEmbedGGUFModel, task="gguf-embedding-feature-extraction"
    ),
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
    "biogpt": ModelRegistration(ScaledEmbeddingGPT2CausalLMModel),
    "ctrl": ModelRegistration(CTRLCausalLMModel),
    "gpt-sw3": ModelRegistration(GPT2CausalLMModel),
    "gpt2": ModelRegistration(GPT2CausalLMModel),
    "gpt_bigcode": ModelRegistration(GPT2CausalLMModel),
    "gpt_neo": ModelRegistration(GPT2CausalLMModel),
    "imagegpt": ModelRegistration(GPT2CausalLMModel),
    "openai-gpt": ModelRegistration(GPT2CausalLMModel),
    "opt": ModelRegistration(OPTCausalLMModel),
    "xglm": ModelRegistration(ScaledEmbeddingGPT2CausalLMModel),
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
    "t5encoder": ModelRegistration(T5EncoderModel, task="t5-text-encoding"),
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
    "mms": ModelRegistration(Wav2Vec2ForCTCModel, task="ctc-asr", config_class=MMSConfig),
    "parakeet_ctc": ModelRegistration(
        ParakeetForCTCModel,
        task="feature-ctc-asr",
        config_class=ParakeetCTCConfig,
    ),
    "fastconformer_rnnt": ModelRegistration(EncDecRNNTModel, task="fastconformer-rnnt"),
    "sortformer": ModelRegistration(SortformerDiarizationModel, task="diarization"),
}


def _create_default_registry() -> ModelRegistry:
    """Create the default registry with all built-in architectures."""
    reg = ModelRegistry()
    for arch, entry in _REGISTRATIONS.items():
        reg.register(
            arch,
            entry.module_class,
            task=entry.task,
            config_class=entry.config_class,
            test_model_id=entry.test_model_id,
            family=entry.family,
            variant=entry.variant,
            test_revision=entry.test_revision,
        )
    # Attach test_model_id, family, and variant metadata to registrations.
    _apply_test_metadata(reg)
    return reg


# -- Text-only export overrides --
# Maps a multimodal ``model_type`` to its text-only registry sibling, used by
# ``build(..., text_only=True)`` to export the text backbone of a multimodal
# checkpoint as a standalone decoder-only LLM (e.g. so it can use
# GroupQueryAttention instead of the multimodal bidirectional float-bias path).
# Entries must point at a registered text-only model_type. The mapping is
# idempotent: a text-only model_type maps to itself so ``text_only=True`` is a
# no-op when the resolved type is already text-only.
_TEXT_ONLY_MODEL_TYPE: dict[str, str] = {
    "muse_glimmer": "muse_glimmer_text",
    "muse_glimmer_text": "muse_glimmer_text",
    "gemma3n": "gemma3n_text",
    "gemma3n_text": "gemma3n_text",
    # Shipped Gemma 4 multimodal checkpoints (e.g. ``google/gemma-4-E2B-it``)
    # declare ``model_type="gemma4"`` with a nested ``text_config`` whose own
    # ``model_type`` is ``gemma4_text``. Both resolve to the same
    # ``Gemma4CausalLMModel`` backbone, so ``text_only=True`` is supported.
    "gemma4": "gemma4_text",
    "gemma4_text": "gemma4_text",
    "gemma4_unified": "gemma4_unified_text",
    "gemma4_unified_text": "gemma4_unified_text",
    # Qwen3.5-MoE-VL (Qwen3.6-35B-A3B): export just the hybrid MoE text
    # backbone as a standalone decoder-only LLM. The builder overrides
    # ``qwen3_5_moe`` -> ``qwen3_5_moe_vl`` when a ``vision_config`` is present,
    # so the text-only override keys off the VL type here.
    "qwen3_5_moe_vl": "qwen3_5_moe_text",
    "qwen3_5_moe_text": "qwen3_5_moe_text",
    # Qwen3.8 Flash-Next is published as a multimodal qwen4_exp composite.
    # text_only=True selects the same exact decoder without the vision stages.
    "qwen4_exp": "qwen4_exp_text",
    "qwen4_exp_text": "qwen4_exp_text",
}


# fmt: off
# -- Test model IDs for L2 architecture validation --
# Each maps a registered model_type to a public HuggingFace model.
# Only the config.json is downloaded (no weights).
_TEST_MODEL_IDS: dict[str, str] = {
    # --- CausalLM (Llama-compatible) ---
    "llama": "meta-llama/Llama-3.2-1B",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "qwen2": "Qwen/Qwen2.5-0.5B",
    "qwen4_exp": "Qwen/Qwen3.8-Flash-Next",
    "qwen4_exp_text": "unsloth/Qwen3.8-Flash-Next-FP8",
    "Qwen4ExpForConditionalGeneration": "Qwen/Qwen3.8-Flash-Next",
    "plamo2": "pfnet/plamo-2-1b",
    "cohere": "CohereForAI/c4ai-command-r7b-12-2024",
    "cohere2": "CohereForAI/c4ai-command-r7b-12-2024",
    "cosmos3_edge": "nvidia/Cosmos3-Edge",
    "cosmos3_edge_text": "nvidia/Cosmos3-Edge",
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
    "bitnet": "microsoft/bitnet-b1.58-2B-4T",
    "talkie": "PocketAiHub/talkie-1930-13b-it-GGUF",
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
    "jais2": "inceptionai/Jais-2-8B-Chat",
    "kclgpt": "WisdomShell/CodeShell-7B",
    "csm": "sesame/csm-1b",
    "evolla": "westlake-repl/Evolla-10B-hf",
    "nemotron_h": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    "nemotron_parse": "nvidia/NVIDIA-Nemotron-Parse-2.0",
    "open-llama": "openlm-research/open_llama_3b",
    "orion": "OrionStarAI/Orion-14B-Base",
    "persimmon": "adept/persimmon-8b-base",
    "shieldgemma2": "google/shieldgemma-2b",
    "solar_open": "upstage/solar-pro-preview-instruct",

    # --- CausalLM (architecture-specific) ---
    "falcon": "tiiuae/falcon-7b",
    "falcon_h1": "tiiuae/Falcon-H1-Tiny-90M-Base",
    "bloom": "bigscience/bloom-560m",
    "gemma": "google/gemma-2b",
    "gemma2": "google/gemma-2-2b",
    "gemma3": "google/gemma-3-4b-it",
    "gemma3_text": "google/gemma-3-1b-pt",
    "gemma_embedding_gguf": "unsloth/embeddinggemma-300m-GGUF",
    # No text-only gemma3n checkpoint was ever published; the -it releases are
    # multimodal, and "gemma3n_text" reaches the text path via _TEXT_ONLY_MODEL_TYPE.
    "gemma3n": "google/gemma-3n-E4B-it",
    "gemma3n_text": "google/gemma-3n-E2B-it",
    "gemma4_text": "google/gemma-4-E2B-it",
    "granite": "ibm-granite/granite-3.3-2b-instruct",
    "internlm2": "internlm/internlm2_5-7b-chat",
    "llama_embed_gguf": "mradermacher/llama-embed-nemotron-8b-GGUF",
    "nemotron": "nvidia/Nemotron-Mini-4B-Instruct",
    "olmo": "allenai/OLMo-1B-hf",
    "olmo2": "allenai/OLMo-2-1124-7B",
    "phi": "microsoft/phi-1_5",
    "phi3": "microsoft/Phi-3.5-mini-instruct",
    "phi3small": "microsoft/Phi-3-small-8k-instruct",
    "qwen": "Qwen/Qwen-1_8B-Chat",
    "qwen3": "Qwen/Qwen3-0.6B",
    "qwen3_5_text": "Qwen/Qwen3.5-2B",
    "Eagle3LlamaForCausalLM": "AngelSlim/Qwen3-4B_eagle3",
    "LlamaForCausalLMEagle3": "AngelSlim/Qwen3-8B_eagle3",
    "smollm3": "HuggingFaceTB/SmolLM3-3B",
    "gpt2": "openai-community/gpt2",
    "opt": "facebook/opt-125m",
    "mpt": "mosaicml/mpt-7b",
    "biogpt": "microsoft/biogpt",
    "chatglm": "zai-org/chatglm2-6b",
    "codegen": "Salesforce/codegen-350M-mono",
    "cosmos3_omni": "nvidia/Cosmos3-Nano",
    "ctrl": "Salesforce/ctrl",
    "ernie4_5": "baidu/ERNIE-4.5-0.3B-PT",
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
    "xverse": "xverse/XVERSE-7B",
    "xlm": "FacebookAI/xlm-mlm-en-2048",

    # --- Mixture of Experts ---
    "mixtral": "mistralai/Mixtral-8x7B-v0.1",
    "phimoe": "microsoft/Phi-tiny-MoE-instruct",
    "phimoe_gguf": "microsoft/Phi-tiny-MoE-instruct",
    "qwen2_moe": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    "qwen3_moe": "Qwen/Qwen3-30B-A3B",
    "qwen3_5_moe": "Qwen/Qwen3.5-MoE-A3B-128K",
    "qwen3_5_moe_text": "Qwen/Qwen3.6-35B-A3B",
    "qwen3_next": "Qwen/Qwen3-235B-A22B",
    "granitemoe": "ibm-granite/granite-3.0-1b-a400m-instruct",
    "olmoe": "allenai/OLMoE-1B-7B-0924",
    "dbrx": "databricks/dbrx-instruct",
    "arctic": "Snowflake/snowflake-arctic-instruct",
    "jetmoe": "jetmoe/jetmoe-8b",
    "kimi_linear": "moonshotai/Kimi-Linear-48B-A3B-Instruct",
    "longcat_flash": "yujiepan/longcat-flash-tiny-random",
    "minimax": "MiniMaxAI/MiniMax-Text-01",
    "ernie4_5_moe": "baidu/ERNIE-4.5-21B-A3B-PT",
    "bailing_moe": "baidu/ERNIE-4.5-21B-A3B-PT",
    "flex_olmo": "allenai/Flex-reddit-2x7B-1T",
    "glm4_moe": "zai-org/GLM-4.5-Air",
    "granitemoehybrid": "ibm-granite/granite-4.0-tiny-preview",
    "granitemoeshared": "ibm-research/moe-7b-1b-active-shared-experts",
    "kimi_k3": "yujiepan/kimi-k3-tiny-random",
    "lfm2_moe": "LiquidAI/LFM2-8B-A1B",
    "MiniMaxText01": "MiniMaxAI/MiniMax-Text-01",
    "minimax_text_01": "MiniMaxAI/MiniMax-Text-01",
    "qwen3_omni_moe": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "qwen3_vl_moe": "Qwen/Qwen3-VL-30B-A3B-Instruct",

    # --- DeepSeek (MLA + MoE) ---
    "deepseek_v2": "deepseek-ai/DeepSeek-V2-Lite",
    "deepseek_v2_moe": "deepseek-ai/DeepSeek-V2-Lite",
    "deepseek_v3": "deepseek-ai/DeepSeek-V3",
    "deepseek": "deepseek-ai/DeepSeek-V3",
    "deepseek_v4": "deepseek-ai/DeepSeek-V4-Flash",
    # --- GLM-5.2 (MLA + DSA + MoE) ---
    "glm_moe_dsa": "zai-org/GLM-5.2",

    # --- SSM (Mamba) ---
    "mamba": "state-spaces/mamba-130m-hf",
    "mamba2": "state-spaces/mamba2-130m",
    "falcon_mamba": "tiiuae/falcon-mamba-7b",

    # --- Hybrid SSM+Attention ---
    "jamba": "ai21labs/Jamba-v0.1",
    "bamba": "ibm-fms/Bamba-9B",
    "lfm2": "LiquidAI/LFM2.5-230M",

    # --- Multimodal ---
    "qwen2_vl": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2_vl_text": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2_5_vl": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2_5_vl_text": "Qwen/Qwen2.5-VL-3B-Instruct",
    "glm_ocr": "zai-org/GLM-OCR",
    "qwen3_vl": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3_vl_text": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3_5": "Qwen/Qwen3.5-2B",
    "llava": "llava-hf/llava-1.5-7b-hf",
    "llava_next": "llava-hf/llava-v1.6-mistral-7b-hf",
    "mllama": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "muse_glimmer": "meta-models/Muse-Glimmer-30B",
    "muse_glimmer_text": "meta-models/Muse-Glimmer-30B",
    "gemma4": "google/gemma-4-E2B-it",
    "gemma4_unified": "google/gemma-4-12B",
    "gemma4_unified_text": "google/gemma-4-12B",
    "internvl2": "OpenGVLab/InternVL2-1B",
    "mage_vl": "microsoft/Mage-VL",
    "phi4mm": "microsoft/Phi-4-multimodal-instruct",
    "phi4_multimodal": "microsoft/Phi-4-multimodal-instruct",
    "phi3_v": "microsoft/Phi-3.5-vision-instruct",
    "phi4-siglip": "microsoft/Phi-4-reasoning-vision-15B",
    "blip-2": "Salesforce/blip2-opt-2.7b",
    "florence2": "microsoft/Florence-2-base",
    "idefics2": "HuggingFaceM4/idefics2-8b",
    "idefics3": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "instructblip": "Salesforce/instructblip-flan-t5-xl",
    "llava_onevision": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    "molmo": "allenai/MolmoE-1B-0924",
    "mistral3": "mistralai/Ministral-3-3B-Instruct-2512",
    "minicpmv4_6": "openbmb/MiniCPM-V-4.6",
    "lfm2_vl": "LiquidAI/LFM2.5-VL-3B",
    "aya_vision": "CohereForAI/aya-vision-8b",
    "chameleon": "facebook/chameleon-7b",
    "cohere2_vision": "CohereForAI/c4ai-command-r7b-12-2024",
    "deepseek_vl": "deepseek-ai/deepseek-vl2-tiny",
    "deepseek_vl_hybrid": "deepseek-ai/deepseek-vl2-tiny",
    "deepseek_vl_v2": "deepseek-ai/deepseek-vl2-tiny",
    "fuyu": "adept/fuyu-8b",
    "glm4v": "THUDM/glm-4v-9b",
    "glm4v_moe": "THUDM/glm-4v-9b",
    "glm4v_moe_text": "zai-org/GLM-4.5V",
    "glm4v_text": "THUDM/glm-4v-9b",
    "got_ocr2": "stepfun-ai/GOT-OCR2_0",
    "hunyuan_vl_mot": "tencent/HY-Embodied-0.5-X",
    "neo_chat": "sensenova/SenseNova-U1.5-8B-MoT",
    "instructblipvideo": "Salesforce/instructblip-flan-t5-xl",
    "internvl": "OpenGVLab/InternVL2-1B",
    "internvl_chat": "OpenGVLab/InternVL-Chat-V1-5",
    "janus": "deepseek-ai/Janus-Pro-1B",
    "llava_next_video": "llava-hf/LLaVA-NeXT-Video-7B-hf",
    "ovis2": "AIDC-AI/Ovis2-1B",
    "paligemma": "google/paligemma-3b-pt-224",
    "pixtral": "mistralai/Pixtral-12B-2409",
    "qwen3_5_moe_vl": "Qwen/Qwen3.6-35B-A3B",
    "qwen3_5_vl": "Qwen/Qwen3.5-2B",
    "qwen3_5_vl_text": "Qwen/Qwen3.5-2B",
    "qwen3_vl_single": "Qwen/Qwen3-VL-2B-Instruct",
    "sam2": "facebook/sam2-hiera-base-plus",
    "smolvlm": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "video_llava": "LanguageBind/Video-LLaVA-7B-hf",
    "vipllava": "llava-hf/vip-llava-7b-hf",

    # --- Speech ---
    "moonshine": "moonshine-ai/moonshine-tiny",
    "whisper": "openai/whisper-tiny",
    "qwen3_asr": "Qwen/Qwen3-ASR-0.6B",
    "fun_asr": "justinchuby/Fun-ASR-Nano-2512",
    "glmasr": "zai-org/GLM-ASR-Nano-2512",
    "sensevoice_small": "mlx-community/SenseVoiceSmall",
    "mms": "facebook/mms-300m",
    "parakeet_ctc": "nvidia/parakeet-ctc-1.1b",
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
    "phi3_v": "phi",
    "phi4-siglip": "phi",
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
    "qwen3_5_moe_text": "qwen",
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
    "qwen3_5_moe_vl": "qwen",
    "qwen3_5_vl": "qwen",
    "qwen4_exp": "qwen",
    "qwen4_exp_text": "qwen",
    "qwen3_omni_moe": "qwen",
    "qwen3_asr": "qwen",
    "qwen3_forced_aligner": "qwen",
    "fun_asr": "qwen",
    "glmasr": "glm",
    "qwen3_tts": "qwen",
    "qwen3_tts_tokenizer_12hz": "qwen",
    "deepseek_v2": "deepseek",
    "deepseek_v2_moe": "deepseek",
    "deepseek_v3": "deepseek",
    "deepseek_v4": "deepseek",
    "deepseek_vl_v2": "deepseek",
    "glm_moe_dsa": "glm",
    "olmo": "olmo",
    "olmo2": "olmo",
    "olmo3": "olmo",
    "olmoe": "olmo",
    "llama": "llama",
    "code_llama": "llama",
    "llama4_text": "llama",
    "lfm2": "lfm",
    "lfm2_vl": "lfm",
    "mllama": "llama",
    "mistral": "mistral",
    "mistral3": "mistral",
    "ministral": "mistral",
    "ministral3": "mistral",
    "mixtral": "mistral",
    "pixtral": "mistral",
    "falcon": "falcon",
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
    "mms": "wav2vec2",
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
    "deepseek_v4": "dense-csa-fallback+mtp+moe+hc",
    "glm_moe_dsa": "mla+dsa-indexshare+full-attention-fallback+moe",
    "phi3small": "blocksparse",
    "mamba": "ssm",
    "mamba2": "ssm",
    "falcon_mamba": "ssm",
    "jamba": "hybrid-ssm+attn",
    "bamba": "hybrid-mamba2+attn",
    "lfm2": "hybrid-conv+attn",
    "lfm2_vl": "siglip2-naflex+hybrid-conv+attn",
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
