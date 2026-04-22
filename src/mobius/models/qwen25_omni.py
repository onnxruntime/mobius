"""Qwen2.5-Omni: Multimodal model with audio + vision + text.

Architecture (Thinker only):
  - Audio encoder: Conv1d x2 → sinusoidal PE → 32 encoder layers → AvgPool → proj
  - Vision encoder: Conv3d patch embed → 32 ViT blocks → patch merger
  - Fusion: Audio/vision features replace placeholder token positions
  - Text decoder: Qwen2 (no QK norm) + MRoPE

Reference: https://huggingface.co/Qwen/Qwen2.5-Omni-7B
HuggingFace class: Qwen2_5OmniForConditionalGeneration
"""


class Qwen25OmniAudioEncoder(nn.Module):
    pass

class Qwen25OmniVisionEncoder(nn.Module):
    pass

class Qwen25OmniEmbeddingModel(nn.Module):
    pass

class Qwen25OmniDecoderModel(nn.Module):
    pass

class Qwen25OmniThinkerForConditionalGeneration(nn.Module):
    pass


