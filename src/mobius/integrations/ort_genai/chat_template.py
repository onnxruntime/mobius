# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""ORT-GenAI chat-template compatibility helpers.

The Gemma-4 fallback template supports ordinary text turns and structured
text/image/audio/video content. It intentionally does not implement tool
definitions, tool calls, or tool responses; packages that require those
branches must provide a separately validated ORT-compatible custom template.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

GEMMA4_ORT_CHAT_TEMPLATE = """{{- bos_token -}}
{%- for message in messages -%}
{{- '<|turn>' + ('model' if message['role'] == 'assistant' else message['role']) + '\\n' -}}
{%- if message['content'] is string -%}
{{- message['content'] | trim -}}
{%- else -%}
{%- for item in message['content'] -%}
{%- if item['type'] == 'text' -%}
{{- item['text'] | trim -}}
{%- elif item['type'] == 'image' -%}
{{- '<|image|>' -}}
{%- elif item['type'] == 'audio' -%}
{{- '<|audio|>' -}}
{%- elif item['type'] == 'video' -%}
{{- '<|video|>' -}}
{%- endif -%}
{%- endfor -%}
{%- endif -%}
{{- '<turn|>\\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|turn>model\\n' -}}
{%- endif -%}
"""

_MULTIMODAL_GEMMA4_MODEL_TYPES = frozenset({"gemma4", "gemma4_unified"})
_STRINGIFIED_MAPPING = re.compile(r"""[\[{]\s*["']type["']\s*:""")


def _is_multimodal_gemma4_model_type(model_type: str) -> bool:
    return model_type in _MULTIMODAL_GEMMA4_MODEL_TYPES


def gemma4_template_needs_ort_normalization(template: str) -> bool:
    """Return whether a Gemma-4 template fails ORT-safe structured-media validation.

    A sandboxed render proves that text, image, and audio inputs produce distinct
    prompts with the required media tokens and no stringified mapping.
    """
    try:
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        compiled = ImmutableSandboxedEnvironment().from_string(template)

        def _render(content: str | list[dict[str, str]]) -> str:
            return compiled.render(
                bos_token="<bos>",
                eos_token="<eos>",
                messages=[{"role": "user", "content": content}],
                tools=None,
                enable_thinking=False,
                preserve_thinking=False,
                add_generation_prompt=True,
            )

        text_render = _render("probe")
        image_render = _render([{"type": "image"}, {"type": "text", "text": "probe"}])
        audio_render = _render([{"type": "audio"}, {"type": "text", "text": "probe"}])
    except Exception:
        logger.debug("Gemma-4 chat-template structural validation failed", exc_info=True)
        return True

    renders = (text_render, image_render, audio_render)
    return not (
        all("probe" in rendered for rendered in renders)
        and "<|image|>" not in text_render
        and "<|audio|>" not in text_render
        and "<|image|>" in image_render
        and "<|audio|>" not in image_render
        and "<|audio|>" in audio_render
        and "<|image|>" not in audio_render
        and len(set(renders)) == len(renders)
        and not any(_STRINGIFIED_MAPPING.search(rendered) for rendered in renders)
    )


def write_chat_template_artifacts(output_dir: str | Path, template: str) -> str:
    """Write one template to both ORT-GenAI tokenizer artifacts.

    ``chat_template.jinja`` is always written. When ``tokenizer_config.json``
    exists, its ``chat_template`` field is updated to the exact same string.
    """
    output_path = Path(output_dir)
    template_path = output_path / "chat_template.jinja"
    template_path.write_text(template, encoding="utf-8")

    tokenizer_config_path = output_path / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        tokenizer_config["chat_template"] = template
        tokenizer_config_path.write_text(
            json.dumps(tokenizer_config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return str(template_path)


def synchronize_chat_template_for_ort(
    output_dir: str | Path,
    model_type: str,
) -> str | None:
    """Synchronize exported chat templates and normalize incompatible Gemma-4 media logic.

    A compatible standalone template is preferred over the JSON copy, preserving
    custom Gemma-4 and non-Gemma templates. If every available Gemma-4 template
    either uses ORT-incompatible structured-media access or fails structural
    text/image/audio rendering validation, both artifacts are replaced with
    the focused text/image/audio/video fallback.
    """
    output_path = Path(output_dir)
    template_path = output_path / "chat_template.jinja"
    tokenizer_config_path = output_path / "tokenizer_config.json"

    file_template = (
        template_path.read_text(encoding="utf-8") if template_path.exists() else None
    )
    tokenizer_config: dict = {}
    if tokenizer_config_path.exists():
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    config_template = tokenizer_config.get("chat_template")
    if not isinstance(config_template, str) or not config_template:
        config_template = None

    templates = [template for template in (file_template, config_template) if template]
    if not templates:
        return None

    selected_template = templates[0]
    normalized = False
    if _is_multimodal_gemma4_model_type(model_type):
        compatible_templates = [
            template
            for template in templates
            if not gemma4_template_needs_ort_normalization(template)
        ]
        if compatible_templates:
            selected_template = compatible_templates[0]
        else:
            selected_template = GEMMA4_ORT_CHAT_TEMPLATE
            normalized = True

    path = write_chat_template_artifacts(output_path, selected_template)
    if normalized:
        logger.warning(
            "Replaced an ORT-incompatible Gemma-4 chat template with the "
            "text/image/audio/video compatibility template. Tool-calling branches "
            "are not supported by this fallback."
        )
    return path
