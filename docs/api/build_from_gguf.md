# `build_from_gguf()`

Build ONNX packages directly from GGUF metadata and tensors without tracing PyTorch.
Support is capability-specific: graph import does not imply runtime packaging.

<!-- BEGIN GGUF CLOSURE SUMMARY -->

**Pinned source:** `ggml-org/llama.cpp@8d9af256337d1a501250f9bbf4c0859a654bddd6` (2026-08-23T16:59:42Z).

| Census | Total | Closure |
|---|---:|---|
| Architectures | 147 | graph verdicts: {'deferred': 55, 'rejected': 2, 'supported': 90}; importable: 89; quantized import: {'rejected': 31, 'supported': 116}; runtime: {'deferred': 142, 'rejected': 2, 'supported': 3} |
| Active stored qtypes | 25 | 24 have an import route; 1 are explicitly deferred with no route |
| Serialized projector strings | 60 | {'graph-importable': 5, 'runtime-supported': 0} |
| Tokenizer pre identifiers | 87 | 56 semantic groups; all default to deferred and become materializable only from a validated embedded `tokenizer.huggingface.json` or exact artifact-scoped tokenizer evidence |

`SUPPORTED` means the named capability is implemented and mechanically tested. `DEFERRED` means it is intentionally unavailable pending the stated work. `REJECTED` means the input or route is invalid by policy. Graph support proves construction/execution only; runtime support additionally requires a pinned real artifact, independent parity, and deterministic generation or stateful semantics. Tokenizer `copy` requires embedded ordered-vocabulary identity; `pinned-source` also binds the complete GGUF artifact, immutable Hub assets, reconstruction policy, semantic hashes, and representative token-ID vectors.

<!-- END GGUF CLOSURE SUMMARY -->

## Usage

```python
from mobius.integrations.gguf import build_from_gguf

package = build_from_gguf("model.gguf")
package.save("output")
```

Use `keep_quantized=False` for explicit float import. Pass `mmproj=` only for an
evidenced multimodal sidecar. The CLI equivalent is `mobius build model.gguf -o output`.

## API

```python
build_from_gguf(
    gguf_path,
    *,
    task=None,
    dtype=None,
    keep_quantized=True,
    execution_provider="default",
    mmproj=None,
    static_cache=False,
    max_seq_len=None,
    allow_dense_moe=None,
    reuse_gguf_weights=False,
    target_config=None,
)
```

The function returns a `ModelPackage`. Import validates architecture metadata, exact tensor
closure, shapes, qtypes, and selected graph route before publication. Source reuse requires
the original immutable GGUF at runtime. Runtime packages additionally require an exact
artifact, graph, tokenizer, runtime version, parity proof, and deterministic state/generation
evidence match.

## Runtime evidence

| Evidence ID | GGUF identity | Config identity | Tokenizer identity | Runtime proof |
|---|---|---|---|---|
| `lfm2-350m-f16-ort-genai-0.15.2` | `LiquidAI/LFM2-350M-GGUF@8fdc9d526b7ed346b19257551b05816c7912ecc2`<br>`LFM2-350M-F16.gguf`<br>711,482,304 B<br>`379ffdcbf08147c0313f6f1ce7ff558a2bc935eda633f4b46c52347032419c42` | `LiquidAI/LFM2-350M@f37d3f5c8c5484bc01dad379a595cf4c68c4e70e` | `LiquidAI/LFM2-350M@73e3c253078a3b97c2e14b4c4665679f4d9b6d56`<br>`chat_template.jinja` 209 B `a805e50fed68938a076b07e2e602639611b50b1ced0e50f11eb92f1ba25be4dc`, `special_tokens_map.json` 434 B `742aefe2b7dec496e8caffdba03a75d0c1a9925d53bd3f3e0d388c96b591b6f4`, `tokenizer.json` 4,732,426 B `98cff83b4f6d7e9d8929bebc62b07e92cf1b3f99c80d16bafe8b84a75448f40b`, `tokenizer_config.json` 91,509 B `36f511115e9d8952cbc9d15d9a20dfa7ce7d1444940e5c1dc42a762020c99bf5`<br>metadata `e5626d605bb50bc53fdb0fbfcf374fb33dfbaa0cc698d9746ba1e9b0b7e6d07d` | ort-genai 0.15.2; full-logit; hybrid convolution and KV state prefill, replay, rollback, reorder, and 20 decode steps |
| `qwen2.5-0.5b-instruct-q8-ort-genai-0.15.2` | `Qwen/Qwen2.5-0.5B-Instruct-GGUF@9217f5db79a29953eb74d5343926648285ec7e67`<br>`qwen2.5-0.5b-instruct-q8_0.gguf`<br>675,710,816 B<br>`ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e` | `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775` | `Qwen/Qwen2.5-0.5B-Instruct@a338b55dd21219a5f4da42bc11a9313d1a27d4cc`<br>`tokenizer.json` 7,031,645 B `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`, `tokenizer_config.json` 7,308 B `5214600ee45ca2f887ce2eede8910378a0111ea99d657428bcbce94778e65a92`<br>metadata `8fc8ef848104e931f14ae03d9581699d54813a2ff952fb7caac0654e8aa27ee3` | ort-genai 0.15.2; full-logit; dynamic KV cache prefill, replay, rollback, reorder, and 20 decode steps |
| `smollm-135m-f16-onnxruntime-1.29.0` | `neopolita/smollm-135m-gguf@22cca988936eafe92908e7558907c3964e10bba7`<br>`ggml-model-f16.gguf`<br>270,885,504 B<br>`ec8c775c16944a7e4b5251f97b3f848500dcc3e701b0d492ce9055cea42138a2` | `HuggingFaceTB/SmolLM-135M@1d461723eec654e65efdc40cf49301c89c0c92f4` | `HuggingFaceTB/SmolLM-135M@1d461723eec654e65efdc40cf49301c89c0c92f4`<br>`special_tokens_map.json` 831 B `e786b595b9a23148bf1630df78d9037a048ea671e48bfd3549a1e3c233742bb3`, `tokenizer.json` 2,104,556 B `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`, `tokenizer_config.json` 3,685 B `238ad6b60d48e471624ea70bc79e92f2611844d5016471fee8c167854bcb98e8`<br>metadata `46646ba36ecae43de6f9f649d217774b889e0fd405af92205319b882927493fc` | onnx-genai 1.29.0; full-logit; dynamic KV cache prefill plus 20 cache-threaded decode steps |
| `smollm-135m-f16-ort-genai-0.15.2` | `neopolita/smollm-135m-gguf@22cca988936eafe92908e7558907c3964e10bba7`<br>`ggml-model-f16.gguf`<br>270,885,504 B<br>`ec8c775c16944a7e4b5251f97b3f848500dcc3e701b0d492ce9055cea42138a2` | `HuggingFaceTB/SmolLM-135M@1d461723eec654e65efdc40cf49301c89c0c92f4` | `HuggingFaceTB/SmolLM-135M@1d461723eec654e65efdc40cf49301c89c0c92f4`<br>`special_tokens_map.json` 831 B `e786b595b9a23148bf1630df78d9037a048ea671e48bfd3549a1e3c233742bb3`, `tokenizer.json` 2,104,556 B `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`, `tokenizer_config.json` 3,685 B `238ad6b60d48e471624ea70bc79e92f2611844d5016471fee8c167854bcb98e8`<br>metadata `46646ba36ecae43de6f9f649d217774b889e0fd405af92205319b882927493fc` | ort-genai 0.15.2; full-logit; ORT GenAI prefill plus 20 cache-threaded decode steps |

Qwen2.5 remains covered only by its existing runtime evidence record above.

## Tokenizer evidence

| Evidence ID | GGUF identity | Official source | Exact tokenizer proof |
|---|---|---|---|
| `qwen3.5-0.8b-q4-tokenizer` | `ggml-org/Qwen3.5-0.8B-GGUF@8fea620810c4afa23dd6443f999a48574c1611a3`<br>`Qwen3.5-0.8B-Q4_0.gguf`<br>563,036,064 B<br>`57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf`<br>320 tensors: F32=133, Q4_0=186, Q8_0=1 | `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`<br>`config.json` 2,907 B `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204`<br>`chat_template.jinja` 7,755 B `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80`, `tokenizer.json` 12,807,982 B `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`, `tokenizer_config.json` 16,709 B `49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c` | GGUF metadata `45302b58b2086a666a874652d0e9e1d5b4b26e786ffbaf9362a4f902eba0b10d`<br>248,320 ordered tokens `5ee0f927bcaa4b9fe85c244776ae9487468e427f83e053fc81f2a186f14936a3`<br>247,587 ordered merges `7e299304d9ad9dc312acdbcb1f6ccf0dce1256bf1aa986d651f13814dfd27e7b`<br>official source IDs `0..248076`; deterministic unused `[PAD{id}]` IDs `248077..248319`; embedding rows=248,320<br>materialized `d91d6b29a588b072bd90f3598ee9097049b8082f0bc43e8a3b41da604bdfe1ee`<br>`Hello, world! 12345` → `[9419, 11, 1814, 0, 220, 16, 17, 18, 19, 20]`<br>`  spaced  text\n` → `[220, 61674, 220, 1414, 198]`<br>`\u4f60\u597d\uff0c\u4e16\u754c\uff01` → `[109266, 3709, 96748, 6115]`<br>`Caf\xe9 \u2014 \u03ba\u03cc\u03c3\u03bc\u03bf\u03c2 \U0001f680` → `[34, 2492, 933, 1892, 166265, 203260, 10838, 248, 222]`<br>`<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n` → `[248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198]`<br>`<|audio_start|><|audio_pad|><|audio_end|>` → `[248070, 248076, 248071]` |

```python
from mobius.integrations.gguf import materialize_evidenced_gguf_tokenizer

materialize_evidenced_gguf_tokenizer("Qwen3.5-0.8B-Q4_0.gguf", "tokenizer")
```

Qwen3.5 tokenizer materialization uses `pinned-source`, not embedded `copy`: official
`tokenizer.json` covers IDs 0-248069, official `tokenizer_config.json` adds IDs
248070-248076, and only the GGUF embedding-alignment suffix 248077-248319 is reconstructed
as non-special unused `[PAD{id}]` tokens. Every ordered token, merge, special ID, source
asset, chat template, representative encoding, and the final materialized hash is checked
before output. Qwen3.5 full model runtime remains **deferred**.

## Supported GGUF architectures

Reason codes are concise user-facing categories; detailed architecture audits remain in
`_arch_registry.py` and its tests.

<!-- BEGIN GGUF SUPPORT MATRIX (generated; see _arch_registry.py) -->

| Canonical architecture | Aliases | Import route | Tensor exactness | Config/tensor/graph/runtime/quantized import | Restriction or evidence gap |
|---|---|---|---|---|---|
| `afmoe` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `apertus` | — | model=`apertus`; tensor=`llama`+`apertus_extras` | exact-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `arcee` | — | model=`arcee`; tensor=`arcee` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `arctic` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `arwkv7` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `baichuan` | — | model=`baichuan`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `bailingmoe` | — | model=`bailing_moe`; tensor=`llama`+`diffusion_fused_qkv`+`moe_extras` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `bailingmoe2` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `bailingmoe3` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `bert` | — | model=`bert`; tensor=`bert` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `bitnet` | — | model=`bitnet`; tensor=`bitnet` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `bloom` | — | model=`bloom`; tensor=`bloom` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `chameleon` | — | none (fails before config extraction) | exact-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `chatglm` | — | model=`chatglm`; tensor=`chatglm` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `clip` | — | none (fails before config extraction) | not claimed | config=rejected; tensor_map=rejected; graph=rejected; runtime=rejected; quantized_import=rejected | CONFIG_REJECTED — The serialized architecture contract is deliberately refused. |
| `codeshell` | — | model=`kclgpt`; tensor=`legacy_layernorm` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `cogvlm` | — | none (fails before config extraction) | exact-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `cohere2` | — | model=`cohere2`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `cohere2moe` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `command-r` | — | model=`command_r`; tensor=`llama`+`command_r_extras` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `dbrx` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `deci` | — | model=`llama`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `deepseek` | — | model=`deepseek`; tensor=`llama`+`diffusion_fused_qkv`+`deepseek_shared_moe_extras` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `deepseek2` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `deepseek2-ocr` | — | none (fails before config extraction) | exact-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `deepseek32` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `deepseek4` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `dflash` | — | model=`DFlashDraftModel`; tensor=`dflash` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `dots1` | — | model=`dots1`; tensor=`llama`+`diffusion_fused_qkv`+`moe_qk_norm_extras`+`deepseek_shared_moe_extras` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `dots3note` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `dream` | — | model=`dream`; tensor=`llama`+`diffusion_fused_qkv` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `eagle3` | — | model=`Eagle3DraftModel`; tensor=`eagle3` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `ernie4_5` | — | model=`ernie4_5`; module=`gguf_legacy`; tensor=`legacy_layernorm` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `ernie4_5-moe` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `eurobert` | — | model=`eurobert`; module=`eurobert_gguf`; tensor=`eurobert` | exact-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `exaone` | — | model=`exaone`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `exaone-moe` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `exaone4` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `falcon` | — | model=`falcon`; tensor=`falcon` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `falcon-h1` | — | model=`falcon_h1`; tensor=`falcon_h1` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `gemma` | — | model=`gemma`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `gemma-embedding` | — | model=`gemma3_text`; module=`gemma_embedding_gguf`; tensor=`gemma_embedding` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `gemma2` | — | model=`gemma2`; tensor=`llama`+`gemma2_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `gemma3` | — | model=`gemma3_text`; tensor=`llama`+`gemma3_extras`; mmproj=`gemma3` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `gemma3n` | — | none (fails before config extraction) | exact-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `gemma4` | — | model=`gemma4_text`; tensor=`llama`+`gemma4_extras`; mmproj=`gemma4` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `gemma4-assistant` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `glm-dsa` | `glm_dsa` | none (no tensor mapping route) | audited-direct-loader-conditional-union | config=supported; tensor_map=deferred; graph=supported; runtime=deferred; quantized_import=supported | TENSOR_MAP_DEFERRED — Exact tensor-name closure is not implemented. |
| `glm4` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `glm4moe` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `gpt-oss` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `gpt2` | — | model=`gpt2`; tensor=`gpt2` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `gptj` | — | none (fails before config extraction) | no-loader | config=rejected; tensor_map=rejected; graph=rejected; runtime=rejected; quantized_import=supported | CONFIG_REJECTED — The serialized architecture contract is deliberately refused. |
| `gptneox` | — | model=`gpt_neox`; module=`gguf_legacy`; tensor=`legacy_layernorm` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `granite` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `granite_swa` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `granitehybrid` | — | model=`granitemoehybrid`; tensor=`granitehybrid` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `granitemoe` | — | model=`granitemoe`; tensor=`llama`+`moe_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `graniteswitch` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `grok` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `grovemoe` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `hunyuan-dense` | `hunyuan_v1_dense` | model=`hunyuan_v1_dense`; tensor=`llama`+`hunyuan_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `hunyuan-moe` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `hunyuan_vl` | — | none (fails before config extraction) | exact-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `hy_v3` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `internlm2` | — | model=`internlm2`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `jais` | — | model=`jais`; module=`gguf_legacy`; tensor=`legacy_layernorm` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `jais2` | — | model=`jais2`; tensor=`legacy_layernorm` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `jamba` | — | model=`jamba`; tensor=`jamba` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `jina-bert-v2` | — | model=`bert`; module=`jina_bert_v2_gguf`; tensor=`jina_bert_v2` | exact-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `jina-bert-v3` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `kimi-k3` | — | model=`kimi_k3`; tensor=`kimi_k3` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `kimi-linear` | — | model=`kimi_linear`; tensor=`kimi_linear` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `laguna` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `lfm2` | — | model=`lfm2`; tensor=`lfm2` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=supported; quantized_import=supported | EVIDENCED_SCOPE — Runtime publication is limited to registry-linked immutable evidence. |
| `lfm2moe` | — | model=`lfm2_moe`; tensor=`lfm2`+`lfm2_moe_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `llada` | — | model=`llada`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `llada-moe` | — | model=`llada`; module=`llada_moe`; tensor=`llama`+`diffusion_fused_qkv`+`moe_qk_norm_extras`+`moe_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `llama` | `mistral` | model=`llama`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=supported; quantized_import=supported | EVIDENCED_SCOPE — Runtime publication is limited to registry-linked immutable evidence. |
| `llama-embed` | — | model=`llama`; module=`llama_embed_gguf`; tensor=`llama_embedding` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `llama4` | — | none (fails before config extraction) | exact-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `maincoder` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `mamba` | — | model=`mamba`; tensor=`mamba` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `mamba2` | — | model=`mamba2`; tensor=`mamba2` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `mellum` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `mimo2` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `minicpm` | — | model=`minicpm`; module=`minicpm_gguf`; tensor=`llama` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `minicpm3` | — | model=`minicpm3`; module=`minicpm3_gguf`; tensor=`minicpm3` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `minimax-01` | — | model=`minimax`; tensor=`minimax` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `minimax-m2` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `minimax-m3` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `mistral3` | — | none (fails before config extraction) | exact-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `mistral4` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `modern-bert` | — | model=`modernbert`; tensor=`modern_bert` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `mpt` | — | model=`mpt`; module=`gguf_legacy`; tensor=`legacy_layernorm` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `muse-glimmer` | `muse_glimmer` | model=`muse_glimmer_text`; tensor=`llama`+`muse_glimmer_extras`; mmproj=`muse_glimmer` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `nanbeige` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `nemotron` | — | model=`nemotron`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `nemotron_h` | — | model=`nemotron_h`; tensor=`nemotron_h` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `nemotron_h_moe` | — | model=`nemotron_h`; tensor=`nemotron_h_moe` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `neo-bert` | — | model=`neobert`; module=`neo_bert_gguf`; tensor=`neo_bert` | exact-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `nomic-bert` | — | model=`nomic_bert`; module=`nomic_bert_gguf`; tensor=`nomic_bert` | exact-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `nomic-bert-moe` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `olmo` | — | model=`olmo`; tensor=`olmo` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `olmo2` | — | model=`olmo2`; tensor=`llama`+`olmo2_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `olmoe` | — | model=`olmoe`; tensor=`llama`+`moe_qk_norm_extras`+`moe_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `openelm` | — | model=`openelm`; module=`gguf_legacy`; tensor=`legacy_layernorm`+`exact_legacy_gguf_extras` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `orion` | — | model=`orion`; tensor=`legacy_layernorm` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `paddleocr` | — | none (fails before config extraction) | strongest-converter-family-inventory-loader-inherited-from-ernie4_5-with-optional-attn-output-bias | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `pangu-embedded` | — | model=`pangu_embedded`; tensor=`llama` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `phi2` | — | model=`phi`; tensor=`phi2` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `phi3` | — | model=`phi3`; tensor=`phi3` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `phimoe` | — | model=`phimoe`; module=`phimoe_gguf`; tensor=`llama`+`phi3`+`moe_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `plamo` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `plamo2` | — | model=`plamo2`; tensor=`plamo2` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `plamo3` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `plm` | — | model=`plm`; tensor=`plm` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `pockettts` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `qwen` | — | model=`qwen`; tensor=`llama`+`qwen1_extras` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `qwen2` | — | model=`qwen2`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=supported; quantized_import=supported | EVIDENCED_SCOPE — Runtime publication is limited to registry-linked immutable evidence. |
| `qwen2moe` | `qwen2_moe` | model=`qwen2_moe`; tensor=`llama`+`moe_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `qwen2vl` | — | model=`qwen2_vl_text`; tensor=`llama`; mmproj=`qwen_vl` | exact-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `qwen3` | — | model=`qwen3`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `qwen35` | — | model=`qwen3_5_text`; tensor=`llama`+`qwen35_hybrid_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `qwen35moe` | — | model=`qwen3_5_moe`; tensor=`llama`+`moe_extras`+`qwen35_hybrid_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `qwen3moe` | `qwen3_moe` | model=`qwen3_moe`; tensor=`llama`+`moe_qk_norm_extras`+`moe_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `qwen3next` | — | model=`qwen3_next`; tensor=`llama`+`moe_extras`+`qwen3next_hybrid_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `qwen3tts` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `qwen3vl` | — | none (fails before config extraction) | exact-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `qwen3vlmoe` | — | none (fails before config extraction) | exact-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `refact` | — | model=`refact`; module=`gguf_legacy`; tensor=`legacy_layernorm` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `rnd1` | — | model=`llada`; module=`rnd1`; tensor=`llama`+`diffusion_fused_qkv`+`moe_qk_norm_extras`+`moe_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `rwkv6` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `rwkv6qwen2` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `rwkv7` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `seed_oss` | — | model=`seed_oss`; tensor=`llama`+`seed_oss_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `smallthinker` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `smollm3` | — | model=`smollm3`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `stablelm` | — | model=`stablelm`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `starcoder` | — | model=`gpt_bigcode`; tensor=`starcoder` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `starcoder2` | — | model=`starcoder2`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `step35` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `t5` | — | model=`t5`; tensor=`t5` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `t5encoder` | — | model=`t5encoder`; tensor=`t5` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime semantics are not yet evidenced. |
| `talkie` | — | model=`talkie`; tensor=`talkie` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |
| `wavtokenizer-dec` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `xverse` | — | model=`xverse`; tensor=`llama` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, and packed quantized import is unavailable. |

<!-- END GGUF SUPPORT MATRIX -->

## Stored quantization types

<!-- BEGIN GGUF QUANTIZATION MATRIX (generated; see _quant_registry.py) -->

| Stored qtype | ID | Projection/output route | Direct exactness | Embedding route | Expert-major route | Non-MatMul route | Runtime |
|---|---:|---|---|---|---|---|---|
| `Q4_0` | 2 | affine repack | exact | affine repack | affine repack | dequantize to float | deferred |
| `Q4_1` | 3 | affine repack | lossy | affine repack | affine repack | dequantize to float | deferred |
| `Q5_0` | 6 | dequantize/requantize | — | dequantize/requantize | dequantize/requantize | dequantize to float | deferred |
| `Q5_1` | 7 | dequantize/requantize | — | dequantize/requantize | dequantize/requantize | dequantize to float | deferred |
| `Q8_0` | 8 | affine repack | exact | affine repack | affine repack | dequantize to float | deferred |
| `Q2_K` | 10 | dequantize/requantize | — | dequantize/requantize | dequantize/requantize | dequantize to float | deferred |
| `Q3_K` | 11 | dequantize/requantize | — | dequantize/requantize | dequantize/requantize | dequantize to float | deferred |
| `Q4_K` | 12 | affine repack | lossy | affine repack | affine repack | dequantize to float | deferred |
| `Q5_K` | 13 | dequantize/requantize | — | dequantize/requantize | dequantize/requantize | dequantize to float | deferred |
| `Q6_K` | 14 | affine repack | lossy | affine repack | affine repack | dequantize to float | deferred |
| `IQ2_XXS` | 16 | native byte-preserved | — | dequantize/requantize | native byte-preserved | dequantize to float | deferred |
| `IQ2_XS` | 17 | native byte-preserved | — | dequantize/requantize | native byte-preserved | dequantize to float | deferred |
| `IQ3_XXS` | 18 | native byte-preserved | — | dequantize/requantize | native byte-preserved | dequantize to float | deferred |
| `IQ1_S` | 19 | native byte-preserved | — | dequantize/requantize | native byte-preserved | dequantize to float | deferred |
| `IQ4_NL` | 20 | native byte-preserved | — | dequantize/requantize | native byte-preserved | dequantize to float | deferred |
| `IQ3_S` | 21 | native byte-preserved | — | dequantize/requantize | native byte-preserved | dequantize to float | deferred |
| `IQ2_S` | 22 | native byte-preserved | — | dequantize/requantize | native byte-preserved | dequantize to float | deferred |
| `IQ4_XS` | 23 | native byte-preserved | — | dequantize/requantize | native byte-preserved | dequantize to float | deferred |
| `IQ1_M` | 29 | native byte-preserved | — | dequantize/requantize | native byte-preserved | dequantize to float | deferred |
| `TQ1_0` | 34 | dequantize/requantize | — | dequantize/requantize | dequantize/requantize | dequantize to float | deferred |
| `TQ2_0` | 35 | dequantize/requantize | — | dequantize/requantize | dequantize/requantize | dequantize to float | deferred |
| `MXFP4` | 39 | native byte-preserved | — | dequantize/requantize | native byte-preserved | dequantize to float | deferred |
| `NVFP4` | 40 | dequantize/requantize | — | dequantize/requantize | dequantize/requantize | dequantize to float | deferred |
| `Q1_0` | 41 | affine repack | exact | affine repack | affine repack | rejected | deferred |
| `Q2_0` | 42 | rejected | — | rejected | rejected | rejected | deferred |

<!-- END GGUF QUANTIZATION MATRIX -->

## Multimodal projector sidecars

| Artifact ID | Immutable sidecar | Bytes | SHA-256 | Projector types |
|---|---|---:|---|---|
| `qwen2-vl-2b-f16` | `ggml-org/Qwen2-VL-2B-Instruct-GGUF@bb307c036e8a1ed7b663bbd0c35b41c4c9294cfd`<br>`mmproj-Qwen2-VL-2B-Instruct-f16.gguf` | 1,331,656,160 | `ecb20cabcdd8dbc277de06bd6eb980aeb2adfaaba9f199a434e328d205675d03` | `qwen2vl_merger` |
| `qwen25-vl-3b-f16` | `ggml-org/Qwen2.5-VL-3B-Instruct-GGUF@5037fcf163dd95d1e41d1974465f0898ed108ca2`<br>`mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf` | 1,338,428,128 | `b9160fe9d814d1fadf68395677468534778b39ac33c2e7561b7b218626e60d5e` | `qwen2.5vl_merger` |
| `gemma3-4b-f16` | `ggml-org/gemma-3-4b-it-GGUF@ab31416aceb30cd095cb34cc27eea120940964e4`<br>`mmproj-model-f16.gguf` | 851,251,104 | `8c0fb064b019a6972856aaae2c7e4792858af3ca4561be2dbf649123ba6c40cb` | `gemma3` |
| `gemma4-e2b-f16` | `unsloth/gemma-4-E2B-it-GGUF@0314792d7f1f7e229411f620751375812bb9faf2`<br>`mmproj-F16.gguf` | 985,654,080 | `337ee849e80b6169ce9d1d573d424fc1653bcafa5f0cb0cbb901beba54f4b41c` | `gemma4v`, `gemma4a` |
| `muse-glimmer-30b-bf16` | `unsloth/Muse-Glimmer-30B-GGUF@faa5b025c584459c13febfa5c59883516710ae39`<br>`mmproj-Muse-Glimmer-30B-BF16.gguf` | 3,849,173,728 | `7aa788cfe25ae5e4bf4837511f64df22cabe595e58223708274a67b3136f53ab` | `muse-glimmer` |

<!-- BEGIN GGUF MMPROJ SUPPORT MATRIX (generated; see _mmproj_registry.py) -->

| Projector string | Modality | Paired text architecture | Metadata/tensor/graph/runtime | Exactness/evidence |
|---|---|---|---|---|
| `adapter` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `cogvlm` | vision | `cogvlm` | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `deepseekocr` | vision | `deepseek2-ocr` | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `deepseekocr2` | vision | `deepseek2-ocr` | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `dots3note_a` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `dots3note_v` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `dots_ocr` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `exaone4_5` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `gemma3` | vision | `gemma3` | metadata=supported; tensor_map=supported; graph=supported; runtime=deferred | artifact pins=`gemma3-4b-f16` |
| `gemma3na` | audio | `gemma3n` | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `gemma3nv` | vision | `gemma3n` | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `gemma4a` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `gemma4ua` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `gemma4uv` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `gemma4v` | vision | `gemma4` | metadata=supported; tensor_map=supported; graph=supported; runtime=deferred | artifact pins=`gemma4-e2b-f16` |
| `glm4v` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `glma` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `granite4_vision` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `granite_speech` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `hunyuanvl` | vision | `hunyuan_vl` | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `idefics3` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `internvl` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `janus_pro` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `kimik25` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `kimivl` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `ldp` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `ldpv2` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `lfm2` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `lfm2a` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `lightonocr` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `llama4` | vision | `llama4` | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `meralion` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `mimo_audio` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `mimovl` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `minicpmv4_6` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `minimax_m3` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `mlp` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `muse-glimmer` | vision | `muse-glimmer` | metadata=supported; tensor_map=supported; graph=supported; runtime=deferred | artifact pins=`muse-glimmer-30b-bf16` |
| `musicflamingo` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `nemotron_v2_vl` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `paddleocr` | vision | `paddleocr` | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `parakeet` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `phi4` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `pixtral` | vision | `deepseek2`, `llama`, `mistral3`, `mistral4` | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `pockettts_gen` | gen.audio | — | metadata=rejected; tensor_map=rejected; graph=rejected; runtime=rejected | CONFIG_REJECTED — The serialized architecture contract is deliberately refused. |
| `pockettts_spkenc` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `qwen2.5o` | audio, vision | `qwen2vl` | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `qwen2.5vl_merger` | vision | `qwen2vl` | metadata=supported; tensor_map=supported; graph=supported; runtime=deferred | artifact pins=`qwen25-vl-3b-f16` |
| `qwen2a` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `qwen2vl_merger` | vision | `qwen2vl` | metadata=supported; tensor_map=supported; graph=supported; runtime=deferred | artifact pins=`qwen2-vl-2b-f16` |
| `qwen3a` | audio | `qwen3vl`, `qwen3vlmoe` | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `qwen3tts_gen` | gen.audio | — | metadata=rejected; tensor_map=rejected; graph=rejected; runtime=rejected | CONFIG_REJECTED — The serialized architecture contract is deliberately refused. |
| `qwen3tts_spkenc` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `qwen3vl_merger` | vision | `qwen35`, `qwen35moe`, `qwen3vl`, `qwen3vlmoe` | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `resampler` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `step3vl` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `ultravox` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `voxtral` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `yasa2` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |
| `youtuvl` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CONFIG_DEFERRED — Exact configuration ownership is not implemented. |

<!-- END GGUF MMPROJ SUPPORT MATRIX -->

## Tokenizer pre-types

The pre-type is never sufficient evidence by itself. Each row defaults to deferred; only an
embedded exact JSON or a complete artifact-scoped evidence record can materialize a tokenizer.

<!-- BEGIN GGUF TOKENIZER PRE SUPPORT MATRIX -->

| Exact identifier | Canonical semantic group | Pinned pre-type | Default route | Exactness/restriction |
|---|---|---|---|---|
| `a.x-4.0` | `gpt-2` | `GPT2` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `afmoe` | `afmoe` | `AFMOE` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `bailingmoe` | `bailingmoe` | `BAILINGMOE` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `bailingmoe2` | `bailingmoe` | `BAILINGMOE` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `bloom` | `bloom` | `BLOOM` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `chameleon` | `chameleon` | `CHAMELEON` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `chatglm-bpe` | `glm4` | `CHATGLM4` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `codeshell` | `codeshell` | `CODESHELL` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `cohere2moe` | `tiny_aya` | `TINY_AYA` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `command-r` | `command-r` | `COMMAND_R` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `dbrx` | `dbrx` | `DBRX` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `deepseek-coder` | `deepseek-coder` | `DEEPSEEK_CODER` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `deepseek-llm` | `deepseek-llm` | `DEEPSEEK_LLM` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `deepseek-r1-qwen` | `qwen2` | `QWEN2` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `deepseek-v3` | `deepseek-v3` | `DEEPSEEK3_LLM` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `default` | `default` | `DEFAULT` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `exaone` | `exaone` | `EXAONE` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `exaone-moe` | `exaone-moe` | `EXAONE_MOE` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `exaone4` | `gpt-2` | `GPT2` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `f2llmv2` | `qwen2` | `QWEN2` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `falcon` | `falcon` | `FALCON` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `falcon-h1` | `llama3` | `LLAMA3` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `falcon3` | `llama3` | `LLAMA3` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `gemma4` | `gemma4` | `GEMMA4` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `gigachat` | `gpt-2` | `GPT2` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `glm4` | `glm4` | `CHATGLM4` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `gpt-2` | `gpt-2` | `GPT2` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `gpt-4o` | `gpt-4o` | `GPT4O` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `gpt3-finnish` | `gpt3-finnish` | `GPT3_FINNISH` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `granite-docling` | `granite-docling` | `GRANITE_DOCLING` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `granite-embed-multi-311m` | `gemma4` | `GEMMA4` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `granite-embed-multi-97m` | `granite-embed-multi-97m` | `GRANITE_EMB_MULTI` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `grok-2` | `grok-2` | `GROK_2` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `hunyuan` | `hunyuan` | `HUNYUAN` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `hunyuan-dense` | `hunyuan-dense` | `HUNYUAN_DENSE` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `jais` | `jais` | `JAIS` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `jais-2` | `jais-2` | `JAIS2` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `jina-de` | `gpt-2` | `GPT2` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `jina-es` | `gpt-2` | `GPT2` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `jina-v1-en` | `jina-v1-en` | `GPT2_ADD_SEP` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `jina-v2-code` | `jina-v1-en` | `GPT2_ADD_SEP` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `jina-v2-de` | `gpt-2` | `GPT2` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `jina-v2-es` | `gpt-2` | `GPT2` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `jina-v5-nano` | `llama3` | `LLAMA3` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `joyai-llm` | `joyai-llm` | `JOYAI_LLM` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `kanana2` | `gpt-4o` | `GPT4O` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `kimi-k2` | `kimi-k2` | `KIMI_K2` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `kormo` | `qwen2` | `QWEN2` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `laguna` | `laguna` | `LAGUNA` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `lfm2` | `llama3` | `LLAMA3` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `llada-moe` | `bailingmoe` | `BAILINGMOE` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `llama-bpe` | `llama3` | `LLAMA3` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `llama-v3` | `llama3` | `LLAMA3` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `llama3` | `llama3` | `LLAMA3` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `llama4` | `gpt-4o` | `GPT4O` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `megrez` | `megrez` | `QWEN2_CLEAN_SPACES` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `mellum` | `gpt-2` | `GPT2` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `mellum2` | `mellum2` | `MELLUM2` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `midm-2.0` | `llama3` | `LLAMA3` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `minerva-7b` | `minerva-7b` | `MINERVA` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `minicpm5` | `minicpm5` | `MINICPM5` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `minimax-m2` | `minimax-m2` | `MINIMAX_M2` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `modern-bert` | `gpt-2` | `GPT2` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `mpt` | `mpt` | `MPT` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `olmo` | `olmo` | `OLMO` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `phi-2` | `gpt-2` | `GPT2` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `pixtral` | `llama3` | `LLAMA3` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `poro-chat` | `poro-chat` | `PORO` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `qwen2` | `qwen2` | `QWEN2` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `qwen35` | `qwen35` | `QWEN35` | `deferred` | ARTIFACT_PINNED — Exact pinned-source materialization is available only for `qwen3.5-0.8b-q4-tokenizer`; every other artifact remains deferred. |
| `refact` | `refact` | `REFACT` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `roberta-bpe` | `jina-v1-en` | `GPT2_ADD_SEP` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `sarvam-moe` | `sarvam-moe` | `SARVAM_MOE` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `seed-coder` | `seed-coder` | `SEED_CODER` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `smaug-bpe` | `smaug-bpe` | `SMAUG` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `smollm` | `smollm` | `SMOLLM` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `solar-open` | `solar-open` | `SOLAR_OPEN` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `stablelm2` | `stablelm2` | `STABLELM2` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `starcoder` | `starcoder` | `STARCODER` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `superbpe` | `superbpe` | `SUPERBPE` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `talkie` | `gpt-4o` | `GPT4O` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `tekken` | `tekken` | `TEKKEN` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `tiny_aya` | `tiny_aya` | `TINY_AYA` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `trillion` | `trillion` | `TRILLION` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `viking` | `viking` | `VIKING` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `whitespace` | `whitespace` | `WHITESPACE` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |
| `youtu` | `youtu` | `YOUTU` | `deferred` | TOKENIZER_EVIDENCE_REQUIRED — Requires embedded exact JSON or artifact-scoped pinned-source evidence. |

<!-- END GGUF TOKENIZER PRE SUPPORT MATRIX -->

## Validation boundary

Normal tests are deterministic and network-free; committed registry records contain compact
immutable identities and semantic hashes. Real-artifact qualification is performed serially
with pinned revisions, full SHA-256 verification, at least twice the artifact size free, and
independent runtime evidence where runtime support is claimed.
