# Qwen4-Exp text core

Mobius implements the text decoder identified by Hugging Face
`model_type=qwen4_exp_text`, including the composite
`Qwen4ExpForConditionalGeneration` route when `text_only=True`.

The implementation is pinned to:

- `Qwen/Qwen3.8-Flash-Next@f5d08274bafd880402bd16f5e3e6c514136ec06c`
- `unsloth/Qwen3.8-Flash-Next-FP8@41cc25fe32cc20053a59c89716196897580cddf6`
- `huggingface/transformers@598d8ba8baaec7fec5a22da0e2844c7bf4ea20e1`

## Exported architecture

The ONNX graph includes the repeating three-linear/one-QSA attention
schedule, full-kernel Gated-DeltaNet convolution state, recurrent delta-rule
state, four-stream gated residual hyper-connections, exact softmax-first
top-k routed MoE plus the sigmoid-gated shared expert, QSA block pooling and
token selection, and PLE hashed n-gram embeddings with their dilated
convolution and token-context states. The pinned BF16 checkpoint keeps
DeltaNet recurrent math and recurrent cache state in float32, while convolution
state, projections, sparse-attention caches, and logits remain in model dtype.

The flattened cache ABI is:

```text
past_position_ids -> present_position_ids
linear layer:
  conv_state, recurrent_state
PLE linear layer:
  conv_state, recurrent_state, ple_conv_state, ple_context
QSA layer:
  key, value, index_key
```

QSA uses standard ONNX operators to reproduce the selected-token mask, then
runs ordinary dense attention under that mask. This is numerically faithful,
including contiguous left padding, but it does not provide the memory savings
of a dedicated sparse-attention runtime kernel.

## Guarded features

The pinned official Transformers implementation explicitly ignores `mtp.*`
checkpoint tensors and does not define the `fc_embedding`/`fc_hidden`
combination equation or a flattened NextN cache ABI. Mobius follows that
ordinary-decoder behavior: it preserves the source MTP metadata, warns while
dropping sidecar-only tensors, and exports the same next-token causal model.
Configurations with dedicated MTP embeddings fail closed because omitting
those embeddings could change the decoder contract. A future standalone
NextN sidecar requires an authoritative execution equation and cache ABI.

FP8/NVFP4 checkpoint lowering, GGUF import, and the multimodal wrapper are
outside this text-core implementation. The nested `qwen4_exp_text`
configuration and architecture registration are present so a later
multimodal wrapper can reuse the existing Qwen3/Qwen3.5 vision and embedding
components without aliasing this decoder.
