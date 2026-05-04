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

2. **ONNX model exported by mobius** with `--runtime ort-genai`:

   ```bash
   mobius build --model <hf-model-id> --dtype f16 \
     --ep default --runtime ort-genai \
     --external-data safetensors --max-shard-size 5GB \
     output/
   ```

   The output directory must contain `genai_config.json` and tokenizer
   files. For single-model exports (text-only LLMs), model files
   (`model.onnx` + external data) are in the output root. For
   multi-model exports (VLMs, ALMs), sub-directories like `decoder/`,
   `embedding/`, `vision_encoder/` contain each model.

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

### Step 3: Copy model files

Copy the entire mobius export output into the model directory. The
simplest approach copies everything:

```bash
cp -r output/* "${CACHE_DIR}/models/Custom/${MODEL_NAME}/"
```

Or copy selectively (handles both single-model and multi-model layouts):

```bash
# Model files — single-model (root-level) and multi-model (sub-dirs)
cp output/model.onnx* "${CACHE_DIR}/models/Custom/${MODEL_NAME}/" 2>/dev/null
cp -r output/decoder/ "${CACHE_DIR}/models/Custom/${MODEL_NAME}/" 2>/dev/null
cp -r output/embedding/ "${CACHE_DIR}/models/Custom/${MODEL_NAME}/" 2>/dev/null
cp -r output/vision_encoder/ "${CACHE_DIR}/models/Custom/${MODEL_NAME}/" 2>/dev/null
cp -r output/audio_encoder/ "${CACHE_DIR}/models/Custom/${MODEL_NAME}/" 2>/dev/null

# Config and tokenizer files (required)
cp output/genai_config.json "${CACHE_DIR}/models/Custom/${MODEL_NAME}/"
cp output/tokenizer* "${CACHE_DIR}/models/Custom/${MODEL_NAME}/"

# Processor configs for VLMs (copy whichever exist)
cp output/image_processor.json "${CACHE_DIR}/models/Custom/${MODEL_NAME}/" 2>/dev/null
cp output/processor_config.json "${CACHE_DIR}/models/Custom/${MODEL_NAME}/" 2>/dev/null
cp output/audio_processor.json "${CACHE_DIR}/models/Custom/${MODEL_NAME}/" 2>/dev/null

# External data shards (single-model layout)
cp output/*.safetensors "${CACHE_DIR}/models/Custom/${MODEL_NAME}/" 2>/dev/null
```

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
# Override with your specific GenAI build
pip install onnxruntime-genai==<version> --force-reinstall --no-deps
```

This replaces the bundled GenAI with your version. Use `--no-deps` to
avoid re-installing Foundry's other dependencies.

### 2. No CUDA EP in pip SDK version

The pip-installed `foundry-local-sdk` does not include CUDA execution
provider support. Models run on CPU by default.

For GPU inference, use the standalone Foundry Local application (not
the pip package) or set up ORT GenAI with CUDA EP directly.

### 3. Multimodal models require GenAI 0.14+

Vision-language and audio-language models need `onnxruntime-genai`
version 0.14 or later for multi-model pipeline support. Check your
Foundry Local's bundled version:

```python
import onnxruntime_genai as og
print(og.__version__)
```

### 4. Custom model discovery

The Python SDK's `FoundryLocalManager.list_models()` uses a catalog
service that may not index custom models in the cache. Use the CLI
or direct HTTP API as a reliable alternative.

### 5. Model type compatibility

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

- **Exporting models:** `.agents/skills/onnx-export-quantization/SKILL.md`
- **ORT GenAI config:** `.agents/skills/ort-genai-config/SKILL.md`
- **Quality checklist (L5):** `.agents/skills/quality-checklist/SKILL.md`
