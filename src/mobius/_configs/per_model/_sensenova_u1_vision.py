# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SenseNova-U1.5 (``neo_chat``) vision extractor hook.

The NEO-unify vision "encoder" has no transformer blocks at all — it is a
patchify conv, a GELU, an interleaved 2-D RoPE, and a merge conv that
projects straight into the LLM hidden size.  The HF ``vision_config``
therefore carries no ``num_hidden_layers`` / ``num_attention_heads`` and
stores ``llm_hidden_size`` / ``downsample_ratio`` as one-element tuples
(an upstream trailing-comma quirk in ``NEOVisionConfig.__init__``).
"""

from __future__ import annotations

from mobius._configs._extractors import register_vision_hook


def _scalar(value, default):
    """Unwrap the upstream one-element tuple/list quirk."""
    if isinstance(value, (tuple, list)):
        value = value[0] if value else None
    return default if value is None else value


@register_vision_hook()
def _sensenova_u1_vision(config, parent_config, model_type: str, fields: dict):
    # Bare hook: ``SenseNovaU1Config`` extracts the text fields from the
    # nested ``llm_config``, so the dispatcher reports ``model_type`` as
    # ``"qwen3"``.  The composite parent is the only reliable signal.
    del model_type
    if getattr(parent_config, "model_type", None) != "neo_chat":
        return None
    vision_source = getattr(parent_config, "vision_config", None) or config
    llm_hidden_size = _scalar(getattr(vision_source, "llm_hidden_size", None), None)
    if llm_hidden_size is None and parent_config is not None:
        llm_config = getattr(parent_config, "llm_config", None)
        llm_hidden_size = getattr(llm_config, "hidden_size", None)
    downsample_ratio = float(_scalar(getattr(vision_source, "downsample_ratio", None), 0.5))
    fields.update(
        hidden_size=int(getattr(vision_source, "hidden_size", None) or 1024),
        patch_size=int(getattr(vision_source, "patch_size", None) or 16),
        in_channels=int(getattr(vision_source, "num_channels", None) or 3),
        # ``downsample_ratio`` 0.5 means a 2x2 patch merge.
        spatial_merge_size=round(1.0 / downsample_ratio),
        out_hidden_size=int(llm_hidden_size or 4096),
        rope_theta=float(getattr(vision_source, "rope_theta_vision", None) or 10_000.0),
        num_position_embeddings=int(
            getattr(vision_source, "max_position_embeddings_vision", None) or 10_000
        ),
        # No transformer stack in the NEO vision tower.
        num_hidden_layers=0,
        num_attention_heads=0,
    )
    return None
