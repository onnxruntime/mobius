---
name: foundry-local
description: >
  Use this skill when deploying mobius-exported ONNX models to
  Microsoft Foundry Local for local inference. Covers custom model
  registration via the cache directory, inference_model.json format,
  CLI and API usage, SDK limitations, and GenAI version compatibility.
---

# Skill: Deploying to Foundry Local

## When to use

Use this skill when:
- Deploying a mobius-exported ONNX model to Foundry Local for local
  inference
- Registering a custom model in the Foundry Local cache
- Writing the `inference_model.json` descriptor for a custom model
- Troubleshooting Foundry Local model loading or compatibility issues
- Accessing the OpenAI-compatible API endpoint for local models

## Prerequisites

1. **Foundry Local SDK:**

   ```bash
   pip install foundry-local-sdk
   ```

   > ⚠️ **WARNING: Dependency override.** `pip install foundry-local-sdk`
   > installs its own versions of `onnxruntime` and
   > `onnxruntime-genai`, which **override any custom builds** you may
   > have installed. If you need a specific ORT or GenAI version (e.g.
   > built from source with CUDA support), you must reinstall your
   > wheels **after** installing foundry-local-sdk:
   >
   > ```bash
   > pip install foundry-local-sdk
   > pip install --force-reinstall <your_ort_wheel>.whl
   > pip install --force-reinstall <your_genai_wheel>.whl
   > ```
   >
   > This works because Foundry's native core symlinks to the
   > Python-installed `.so` files — replacing them takes effect
   > immediately.

2. **ONNX model exported by mobius** with `--runtime ort-genai`:

   ```bash
   mobius build --model <hf-model-id> --dtype f16 \
     --ep default --runtime ort-genai \
     --external-data safetensors --max-shard-size 5GB \
     output/
   ```

   The output is a Model Package directory containing
   `manifest.json`, a `configs/` subdirectory (genai_config + tokenizer +
   processor configs), and one `<component>/base/{variant.json,
   model.onnx, model.onnx.data.safetensors}` per component
   (`decoder` always, plus `embedding` / `vision_encoder` /
   `audio_encoder` for multimodal exports).

   > **Foundry Local consumes a flat layout today.** The Model Package
   > emitted by mobius needs to be flattened (or the Foundry packager
   > extended). The Step 3 commands below do the flattening.

## Custom model registration

Foundry Local discovers models from its cache directory. To deploy a
custom model, place it in the cache with an `inference_model.json`
descriptor.

### Step 1: Find the cache directory

```bash
foundry cache cd
```

This prints the cache path — typically `~/.foundry/cache/` on Linux
or `%LOCALAPPDATA%\foundry\cache\` on Windows.

### Step 2: Create the model directory

```bash
CACHE_DIR=$(foundry cache cd)
MODEL_NAME="my-custom-model"
mkdir -p "${CACHE_DIR}/models/Custom/${MODEL_NAME}"
```

### Step 3: Copy model files (flatten Model Package → Foundry layout)

```bash
DST="${CACHE_DIR}/models/Custom/${MODEL_NAME}"

# Configs (genai_config.json, tokenizer*, processor configs) live under configs/
cp output/configs/* "${DST}/"

# Per-component model files: <component>/base/{model.onnx,*.safetensors}
# Foundry expects them at the top level (single-component) or in
# <component>/ subdirectories (multi-component).
for comp in decoder embedding vision_encoder audio_encoder; do
  if [ -d "output/${comp}/base" ]; then
    if [ "$(ls -A output)" = "decoder" ] || [ ! -d "output/embedding" ]; then
      # Single-component LLM: flatten model files into the model root
      cp output/${comp}/base/* "${DST}/"
    else
      # Multi-component: keep one subdir per component
      mkdir -p "${DST}/${comp}"
      cp output/${comp}/base/* "${DST}/${comp}/"
    fi
  fi
done
```

> Most users will simplify the above with their own packaging script.
> The key invariant is: Foundry Local needs `genai_config.json`,
> tokenizer files, and `model.onnx` (+ external data) reachable from
> the model directory using the legacy flat layout.

### Step 4: Create `inference_model.json`

Create an `inference_model.json` file in the model directory:

```bash
cat > "${CACHE_DIR}/models/Custom/${MODEL_NAME}/inference_model.json" << 'EOF'
{
  "Name": "my-custom-model",
  "PromptTemplate": {
    "system": "<|im_start|>system\n{Content}<|im_end|>\n",
    "user": "<|im_start|>user\n{Content}<|im_end|>\n",
    "assistant": "<|im_start|>assistant\n{Content}<|im_end|>\n",
    "prompt": "{Messages}<|im_start|>assistant\n"
  }
}
EOF
```

### Step 5: Verify registration

```bash
foundry cache ls
```

Your model should appear in the list.

## `inference_model.json` format

The `inference_model.json` descriptor tells Foundry Local how to
identify and prompt the model:

```json
{
  "Name": "<model-name>",
  "PromptTemplate": {
    "system": "<system-message-template>",
    "user": "<user-message-template>",
    "assistant": "<assistant-message-template>",
    "prompt": "<full-prompt-template>"
  }
}
```

### Fields

| Field | Description |
|-------|-------------|
| `Name` | Display name for the model. Used in CLI and API. |
| `PromptTemplate.system` | Template for system messages. `{Content}` is replaced with the message text. |
| `PromptTemplate.user` | Template for user messages. `{Content}` is replaced with the message text. |
| `PromptTemplate.assistant` | Template for assistant messages. `{Content}` is replaced with the message text. |
| `PromptTemplate.prompt` | Template for the full prompt. `{Messages}` is replaced with the concatenated formatted messages. |

### Common prompt templates

**ChatML (Qwen, many recent models):**

```json
{
  "system": "<|im_start|>system\n{Content}<|im_end|>\n",
  "user": "<|im_start|>user\n{Content}<|im_end|>\n",
  "assistant": "<|im_start|>assistant\n{Content}<|im_end|>\n",
  "prompt": "{Messages}<|im_start|>assistant\n"
}
```

**Llama-style:**

```json
{
  "system": "<|start_header_id|>system<|end_header_id|>\n\n{Content}<|eot_id|>",
  "user": "<|start_header_id|>user<|end_header_id|>\n\n{Content}<|eot_id|>",
  "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{Content}<|eot_id|>",
  "prompt": "<|begin_of_text|>{Messages}<|start_header_id|>assistant<|end_header_id|>\n\n"
}
```

**Gemma-style:**

```json
{
  "system": "",
  "user": "<start_of_turn>user\n{Content}<end_of_turn>\n",
  "assistant": "<start_of_turn>model\n{Content}<end_of_turn>\n",
  "prompt": "{Messages}<start_of_turn>model\n"
}
```

Match the template to your model's chat format. Check the HuggingFace
`tokenizer_config.json` for the model's `chat_template` to determine
the correct format.

## Running the model

### CLI

```bash
foundry model run my-custom-model
```

This starts a local inference server and opens an interactive chat.

### API access

Foundry Local exposes an **OpenAI-compatible API** endpoint. Once a
model is running, you can access it via HTTP:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:5273/v1",
    api_key="foundry-local",  # any non-empty string
)

response = client.chat.completions.create(
    model="my-custom-model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    max_tokens=100,
)
print(response.choices[0].message.content)
```

The port (default 5273) may vary — check the Foundry Local output
when starting the server.

### Python SDK

```python
from foundry_local import FoundryLocalManager

manager = FoundryLocalManager()
# List available models
print(manager.list_models())
```

> **Limitation:** The `FoundryLocalManager` SDK may not discover custom
> models placed in the cache directory. If your model doesn't appear,
> use the CLI (`foundry model run`) or access the HTTP API directly.

## Known limitations

### 1. Bundled GenAI version

Foundry Local bundles its own version of `onnxruntime-genai`. Custom
models must be compatible with the bundled GenAI version. If your model
uses features from a newer GenAI version (e.g. new model_type support),
it may not load.

**Workaround:** Force-reinstall your target GenAI wheel after installing
the Foundry Local SDK:

```bash
pip install foundry-local-sdk
# Override with your specific GenAI build (and ORT if needed)
pip install --force-reinstall <your_ort_wheel>.whl
pip install --force-reinstall <your_genai_wheel>.whl
```

This replaces the bundled GenAI with your version. Foundry's native
core symlinks to the Python-installed `.so` files, so replacing them
takes effect immediately.

### 2. Building ORT and GenAI from source

If you need a custom ORT GenAI build (e.g. with CUDA support, or with
a new model_type registered), build both from source:

1. **Build ORT** with CUDA EP enabled
2. **Build GenAI** linked against your custom ORT build
3. Install both wheels, then install `foundry-local-sdk`, then
   **reinstall your custom wheels** (see warning above)

See the **building-ort-genai** skill
(`.agents/skills/building-ort-genai/SKILL.md`) for the full
step-by-step guide, and
[issue #245](https://github.com/onnxruntime/mobius/issues/245) for a
complete end-to-end tutorial.

### 3. No CUDA EP in pip SDK version

The pip-installed `foundry-local-sdk` does not include CUDA execution
provider support. Models run on CPU by default.

For GPU inference, use the standalone Foundry Local application (not
the pip package) or set up ORT GenAI with CUDA EP directly.

### 4. Multimodal models require GenAI 0.14+

Vision-language and audio-language models need `onnxruntime-genai`
version 0.14 or later for multi-model pipeline support. Check your
Foundry Local's bundled version:

```python
import onnxruntime_genai as og
print(og.__version__)
```

### 5. Custom model discovery

The Python SDK's `FoundryLocalManager.list_models()` uses a catalog
service that may not index custom models in the cache. Use the CLI
or direct HTTP API as a reliable alternative.

### 6. Model type compatibility

Foundry Local uses ORT GenAI internally, which has a model_type
whitelist (see the `ort-genai-config` skill). If your model uses an
unregistered model_type, set `"type": "decoder"` under the `"model"`
key in `genai_config.json` as a workaround:

```json
{
  "model": {
    "type": "decoder"
  }
}
```

## End-to-end example

Complete workflow from HuggingFace model to local inference:

```bash
# 1. Export with mobius
mobius build --model Qwen/Qwen2.5-7B-Instruct \
  --dtype f16 --ep default --runtime ort-genai \
  --external-data safetensors --max-shard-size 5GB \
  qwen2.5-7b/

# 2. Register in Foundry Local cache
CACHE_DIR=$(foundry cache cd)
mkdir -p "${CACHE_DIR}/models/Custom/qwen2.5-7b"
cp -r qwen2.5-7b/* "${CACHE_DIR}/models/Custom/qwen2.5-7b/"

# 3. Create inference_model.json
cat > "${CACHE_DIR}/models/Custom/qwen2.5-7b/inference_model.json" << 'EOF'
{
  "Name": "qwen2.5-7b",
  "PromptTemplate": {
    "system": "<|im_start|>system\n{Content}<|im_end|>\n",
    "user": "<|im_start|>user\n{Content}<|im_end|>\n",
    "assistant": "<|im_start|>assistant\n{Content}<|im_end|>\n",
    "prompt": "{Messages}<|im_start|>assistant\n"
  }
}
EOF

# 4. Verify and run
foundry cache ls
foundry model run qwen2.5-7b
```

## Reference

- [Deploying Custom Models with Olive and Foundry Local](https://techcommunity.microsoft.com/blog/educatordeveloperblog/deploying-custom-models-with-microsoft-olive-and-foundry-local/4489002)

## Cross-references

- **Building ORT/GenAI:** `.agents/skills/building-ort-genai/SKILL.md`
- **Exporting models:** `.agents/skills/onnx-export-quantization/SKILL.md`
- **ORT GenAI config:** `.agents/skills/ort-genai-config/SKILL.md`
- **Quality checklist (L5):** `.agents/skills/quality-checklist/SKILL.md`
