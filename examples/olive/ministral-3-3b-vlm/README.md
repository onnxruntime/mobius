# Ministral-3-3B VLM: E2E Export & Inference Demo

This example demonstrates how to export
[Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512)
vision-language model to ONNX using
[mobius](https://github.com/onnxruntime/mobius) and run inference
with ONNX Runtime GenAI.

Ministral-3-3B is a multimodal (VLM) model combining a Pixtral
vision encoder with a Mistral text decoder. All three sub-models
are exported via mobius, with optional Olive quantization for the
text decoder.

## Prerequisites

Install mobius from the repo root:

```bash
pip install -e '.[transformers]'
```

Install additional dependencies:

```bash
pip install -r requirements.txt
```

Install ONNX Runtime GenAI:

| Device | Install Command |
|--------|------------------|
| CPU | `pip install onnxruntime-genai` |
| GPU | `pip install onnxruntime-genai-cuda` |

Optional — for INT4/FP16 text decoder quantization:

```bash
pip install olive-ai
```

## Steps

### 1. Export Models

**Pure mobius (FP16, all 3 models):**

```bash
python optimize.py
```

**With Olive INT4 quantization for text decoder (CPU):**

```bash
python optimize.py --olive-config cpu_and_mobile/text.json
```

**With Olive FP16 for text decoder (CUDA):**

```bash
python optimize.py --olive-config cuda/text.json --ep cuda
```

**Custom options:**

```bash
python optimize.py --output-dir output/ministral3 --dtype f32
```

This runs:
- **Mobius** for all 3 ONNX models (text decoder, vision
  encoder, embedding) with pretrained weights
- **Mobius genai integration** for `genai_config.json`,
  `processor_config.json`, and tokenizer files
- **(Optional) Olive** for text decoder quantization
  (INT4/FP16 with GQA)

### 2. Output Structure

```
models/
├── decoder/
│   ├── model.onnx              # Text decoder
│   └── model.onnx.data
├── vision/
│   ├── model.onnx              # Pixtral vision encoder
│   └── model.onnx.data
├── embedding/
│   ├── model.onnx              # Embedding fusion model
│   └── model.onnx.data
├── genai_config.json           # Runtime configuration
├── processor_config.json       # Image preprocessing
├── tokenizer.json
└── tokenizer_config.json
```

### 3. Run Inference

```bash
# Text-only
python inference.py --prompt "What is the capital of France?"

# Image + text
python inference.py --image photo.jpg --prompt "Describe this"

# Interactive mode
python inference.py --interactive
```

### 4. Evaluate on AI2D

```bash
# ONNX only (default: 100 samples)
python eval.py --model_path models

# Both ONNX and PyTorch side-by-side
python eval.py --pytorch_model mistralai/Ministral-3-3B-Instruct-2512
```

## Notes

- HuggingFace checkpoint uses FP8 quantized weights. Mobius
  dequantizes automatically during weight loading.
- The tokenizer class (`TokenizersBackend`) is automatically
  remapped to `LlamaTokenizer` by the genai integration.
- Pixtral vision supports dynamic image sizes (multiples of
  28, up to 1540x1540).

## References

- [Pixtral/Ministral3 VLM support (PR #130)](https://github.com/onnxruntime/mobius/pull/130)
- [onnxruntime-genai PR #2077](https://github.com/microsoft/onnxruntime-genai/pull/2077)
