# Experimental ONNX KV-cache example

This example exports pretrained Mobius decoder graphs, converts each layer's
tensor K/V plumbing to the experimental ONNX `kv_cache` type and
`AttentionWithCache` operator, then compares the original and converted graphs
on CPU. The cache backend is selected when creating the runtime value; it is not
encoded in the graph. This is a prototype, not a production serving stack.

## Models and artifacts

| Model | Hugging Face revision | Attention layout |
| --- | --- | --- |
| `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | 28 full-attention layers |
| `google/gemma-3-270m` | `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1` | 15 sliding-window layers (`W=512`), 3 full-attention layers |

The generated files from the verified model build are outside the repository
under:

```text
C:\Users\grama\.copilot\session-state\5be4db02-3ca8-4dea-8bfb-45ac0a0d6d9f\files\kvcache\
  qwen3-0.6b\model.onnx
  qwen3-0.6b\model.onnx.data
  qwen3-0.6b\model.kv_cache.onnx
  qwen3-0.6b\config.json
  qwen3-0.6b\run_results.json
  gemma-3-270m\model.onnx
  gemma-3-270m\model.onnx.data
  gemma-3-270m\model.kv_cache.onnx
  gemma-3-270m\config.json
  gemma-3-270m\run_results.json
```

The `.data` files contain the pretrained weights and are intentionally not
checked in. The build script pins both model revisions, uses Mobius's
unfused/default execution-provider path, saves the external weights, and
validates the baseline and converted graphs. Hugging Face's normal cache is
used, so existing checkpoint files are reused. Recreate artifacts in a chosen
directory with:

```powershell
python examples\kvcache\build_model.py --model Qwen/Qwen3-0.6B --output-dir C:\models\qwen3-0.6b
python examples\kvcache\build_model.py --model google/gemma-3-270m --output-dir C:\models\gemma-3-270m
```

## Graph transformation

The baseline graphs use ONNX `Attention` (default-domain opset 24) with flat
`BS(H*D)` Q/K/V inputs. Qwen3 has 28 Attention nodes and Gemma 3 has 18. Their
serialized graph signatures are:

| Graph | Inputs | Outputs |
| --- | --- | --- |
| Baseline (Qwen 59 / Gemma 39) | `input_ids`, `attention_mask`, `position_ids`, and `past_key_values.<layer>.key` / `.value` for every layer | `logits`, and `present.<layer>.key` / `.value` for every layer (Qwen 57 / Gemma 37) |
| Converted (Qwen 31 / Gemma 21) | `input_ids`, `attention_mask`, `position_ids`, and `past_cache_<layer>` for every layer | `logits` and `present_cache_<layer>` for every layer (Qwen 29 / Gemma 19) |

The converter replaces each tensor-cache pair with one abstract,
runtime-owned cache value input and output per layer, converts Q/K/V to the
operator's BNSD layout, and preserves the original mask, absolute positions,
projections, Qwen3's Q/K normalization, RoPE, grouped-query head mapping,
causal semantics, and per-layer attention geometry.
The external `attention_mask` is the original INT64 logical token mask with
shape `[B,past_tokens + query_tokens]`. The graph constructs the rank-4
Attention mask from it: BOOL for Qwen3 and FLOAT additive bias for Gemma 3;
both use the complete logical key length on the last axis, independent of the
sliding cache's retained storage.

The converted models use default-domain opset 29 and IR version 15. Each cache
carries batch, KV-head, key/value head-size, element-type, and sliding-window
metadata. The nodes are
`AttentionWithCache(Q_new, K_new, V_new, cache, attention_mask, positions) ->
(Y, cache_next)`. Qwen uses `sliding_window=0` for all layers. Gemma uses
`sliding_window=512` for its 15 local-attention layers and `0` for its 3
global-attention layers.

Complete node inventories from the serialized graphs are listed below. The
default ONNX domain is written as `ai.onnx`; the model-local Microsoft
normalization function remains in `com.microsoft`.

| Model | Baseline graph, 574 / 532 nodes |
| --- | --- |
| Qwen3-0.6B | `Attention=28, Cast=1, Concat=1, Expand=1, Gather=3, MatMul=197, Mul=28, RMSNormalization=57, Reshape=112, RotaryEmbedding=56, Shape=3, Swish=28, Transpose=1, Unsqueeze=2, com.microsoft::SkipSimplifiedLayerNormalization=56` |
| Gemma 3-270M | `Add=109, And=3, Attention=18, Cast=1, CumSum=1, Gather=5, Gelu=18, GreaterOrEqual=1, Less=1, MatMul=127, Mul=19, RMSNormalization=73, Reshape=72, RotaryEmbedding=36, Shape=2, Slice=1, Sub=2, Unsqueeze=5, Where=2, com.microsoft::SkipSimplifiedLayerNormalization=36` |

| Model | Converted graph, 798 / 676 nodes |
| --- | --- |
| Qwen3-0.6B | `AttentionWithCache=28, Cast=1, Concat=1, Expand=1, Gather=3, MatMul=197, Mul=28, RMSNormalization=57, Reshape=224, RotaryEmbedding=56, Shape=3, Swish=28, Transpose=113, Unsqueeze=2, com.microsoft::SkipSimplifiedLayerNormalization=56` |
| Gemma 3-270M | `Add=109, And=3, AttentionWithCache=18, Cast=1, CumSum=1, Gather=5, Gelu=18, GreaterOrEqual=1, Less=1, MatMul=127, Mul=19, RMSNormalization=73, Reshape=144, RotaryEmbedding=36, Shape=2, Slice=1, Sub=2, Transpose=72, Unsqueeze=5, Where=2, com.microsoft::SkipSimplifiedLayerNormalization=36` |

Conversion intentionally fails closed if cache tensors are consumed outside
their layer's Attention node, diagnostic Attention outputs are present, head
geometry is inconsistent, or an unsupported Attention attribute would be
lost. The protobuf-level rewrite is isolated in `convert_to_kv_cache.py`
because the experimental cache type is not yet representable through the
stable ONNX IR API.

## Runtime and comparison

The graph conversion requires the experimental ONNX API from
`gramalingam/onnx` commit
`4443045c53d13a8d3157fcbd2cd5e52450a0784b` (including
`helper.make_kv_cache_type` and the opset-29 `AttentionWithCache` schema).
The verified CPython 3.12+ Windows wheel is stored outside the repository at
`C:\Users\grama\.copilot\session-state\51a9868f-d303-40e4-8e9f-da4e7b7a935b\files\onnx-1.24.0-cp312-abi3-win_amd64.whl`
(SHA256
`03029ba7a488ee71bc1ad2f4647a695097e1eca95fb56000e52c89218482f744`).
Install it with:

```powershell
python -m pip install "C:\Users\grama\.copilot\session-state\51a9868f-d303-40e4-8e9f-da4e7b7a935b\files\onnx-1.24.0-cp312-abi3-win_amd64.whl"
```

It was built in Release with Visual Studio 2022 CMake 3.31.6; plain editable
installation is not verified on Windows. The ONNX and ONNX Runtime commits
used below are currently local and unpublished. The source checkouts shown in
the commands are local worktrees; remote clones will not contain these pins
until publication is separately authorized and completed.

To reproduce that Release wheel from the pinned source checkout, ensure CMake
3.31.6 is on `PATH`, map the checkout to a short drive to avoid MSBuild's
260-character path limit, then run:

```powershell
git clone C:\repos\copilot-worktrees\onnx\gramalingam-bookish-broccoli C:\src\onnx-kvcache
Set-Location C:\src\onnx-kvcache
git checkout 4443045c53d13a8d3157fcbd2cd5e52450a0784b
subst K: C:\src\onnx-kvcache
Set-Location K:\
python -m pip wheel . --no-deps --wheel-dir K:\.setuptools-cmake-build-kvcache-release\wheelhouse `
  --config-settings=build-dir=K:\.setuptools-cmake-build-kvcache-release `
  --config-settings=cmake.define.ONNX_BUILD_CUSTOM_PROTOBUF=ON `
  --config-settings=cmake.define.FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER `
  --config-settings=cmake.define.CMAKE_CONFIGURATION_TYPES=Release `
  --config-settings=cmake.define.CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL
```

The native ONNX Runtime fork can be built from its source pin with CPython
3.13, Visual Studio 2022 C++ tools/Windows SDK, Git, CMake, and Ninja available:

```powershell
git clone C:\repos\copilot-worktrees\onnxruntime\gramalingam-cuddly-adventure C:\src\onnxruntime-kvcache
Set-Location C:\src\onnxruntime-kvcache
git checkout 11a522b835ee41b53c6d293a35c7c7135e2dfd5d
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install cmake ninja
python tools\kvcache\build.py --onnx-source C:\src\onnx-kvcache
```

The wrapper checks out the pinned ONNX source into its private build tree,
builds Release with source Protobuf and contrib kernels enabled, and produces
the native test executable and Python wheel. To run the cache unit suite from
the Release output directory:

```powershell
Set-Location build\kvcache-native\Release\Release
.\onnxruntime_kvcache_test.exe --gtest_filter=KVCacheTest.*
```

The verified build used MSVC 14.44.35207, Windows SDK 10.0.26100, CMake 4.4.3,
and two parallel jobs. Its source revision and wheel hash are recorded below.
Install the compatible ONNX and ORT wheels together with
`python -m pip install --force-reinstall <onnx-wheel> <ort-wheel>`.

Mobius itself can be installed in development mode with
`pip install -e ".[transformers,testing]"`.

Use the matching experimental ONNX and ONNX Runtime builds for default-domain
opset 29 and native `AttentionWithCache` support. The native runtime source pin
is `microsoft/onnxruntime` commit
`11a522b835ee41b53c6d293a35c7c7135e2dfd5d` (parent
`31d93935030587d7c362d4e13b00fa5943db851a`). It registers the required
Cast-28 and Swish-24 kernels. The verified CPU-only CPython 3.13 wheel is
`C:\repos\copilot-worktrees\onnxruntime\gramalingam-cuddly-adventure\build\kvcache-native\Release\Release\dist\onnxruntime-1.32.0-cp313-cp313-win_amd64.whl`
(SHA256
`ab1fc5ef214e69ce17ee566fec58ffe1a4a8bf4d0965cc97513cc1c9cd0205d4`).
The runtime's focused native suite passes all 12 `KVCacheTest.*` tests; both
Python smoke tests also pass, including opset-29 Swish-24. The driver enables
experimental opsets by setting `ALLOW_RELEASED_ONNX_OPSET_ONLY=0` before
importing ONNX Runtime and retains contrib kernels required by
`com.microsoft::SkipSimplifiedLayerNormalization`. It allocates runtime-owned
state with `OrtValue.kv_cache(batch_size, num_kv_heads, key_head_size,
value_head_size, backend=..., page_size=16, sliding_window=...)`, inspects
`computed_tokens`, `retained_begin`, and `backend` through `kv_cache_describe()`,
and passes/receives cache OrtValues through `run_with_ort_values`. The runtime
backends are `contiguous`, `paged`, or `ring`; the driver's `mixed` option
selects `paged` for full-attention layers and `ring` for local layers.

Run the real-weight comparison after installing that runtime build:

```powershell
python examples\kvcache\run_example.py --model Qwen/Qwen3-0.6B --model-dir C:\models\qwen3-0.6b
python examples\kvcache\run_example.py --model google/gemma-3-270m --model-dir C:\models\gemma-3-270m
```

The driver runs the baseline tensor-cache graph against contiguous and paged
cache storage, compares complete logits at `atol=rtol=1e-4`, checks greedy
token agreement and cache transitions, and prints generated token IDs/text.
The same serialized converted graph is used for each backend; only runtime
cache allocation selects contiguous, paged, or ring storage.
Gemma defaults to prefill chunks `500,24,3`, deliberately crossing the 512-token
local-attention boundary. Its `mixed` backend uses paged storage for global
layers and a bounded ring cache for local layers. Masks and position IDs stay
in absolute logical-token coordinates even when the ring cache evicts old
entries; the mask key axis remains the complete logical history rather than the
ring's retained physical capacity.

The driver writes machine-readable `run_results.json` next to each model by
default. It records model/runtime/ONNX pins and hashes, prompt and all input
token IDs, exact run arguments, tolerance, per-backend/per-step logit errors
and cache states, and generated tokens. Override the report location with
`--report <path>`.

### Pretrained Qwen result

On the pinned Qwen3 checkpoint, the driver completed an 8-token prefill plus
three decode steps on contiguous and paged storage. Every full-logit comparison
passed `atol=rtol=1e-4` and greedy tokens matched the tensor-cache graph.
Maximum absolute errors for positions `0:8`, `8:9`, `9:10`, and `10:11` were
`9.632111e-5`, `3.433228e-5`, `2.789497e-5`, and `1.907349e-5` for both
backends. `computed_tokens` advanced through `8, 9, 10, 11`; all 28 layers
retained from position 0. The generated token IDs were `[5562, 13, 576]`
(decoded text: ` dog. The`). The full report is beside the model at
`C:\Users\grama\.copilot\session-state\5be4db02-3ca8-4dea-8bfb-45ac0a0d6d9f\files\kvcache\qwen3-0.6b\run_results.json`.

### Pretrained Gemma result (mixed sliding/full attention)

On the pinned Gemma checkpoint, the driver completed chunks `500,24,3`
(527 prefill tokens) plus three decode steps on contiguous, paged, and mixed
(paged full layers/ring sliding layers) storage. All logits passed
`atol=rtol=1e-4` and all greedy tokens agreed with the tensor-cache graph. The
maximum absolute logit errors across the three backends were `5.531311e-5`,
`2.861023e-5`, `2.861023e-5`, `1.335144e-5`, `2.288818e-5`, and `1.406670e-5`
for positions `0:500`, `500:524`, `524:527`, and the three decode steps,
respectively. Cache `computed_tokens` advanced through `500, 524, 527, 528,
529, 530`; sliding-layer `retained_begin` advanced through `0, 12, 15, 16, 17,
18`, while full-layer `retained_begin` remained `0`. The generated token IDs
were `[506, 31770, 4799]` (decoded text: ` the lazy dog`).
The complete command inputs, all 527 token IDs, ONNX/runtime/model hashes,
per-backend step errors, cache states, and generated IDs are recorded beside
the pretrained artifacts in
`C:\Users\grama\.copilot\session-state\5be4db02-3ca8-4dea-8bfb-45ac0a0d6d9f\files\kvcache\gemma-3-270m\run_results.json`.
Token construction is deterministic from the pinned tokenizer and prompt; no
random seed is used.

Focused converter tests are in `tests/kvcache_converter_test.py` and require
the experimental ONNX fork; run them with
`python -m pytest tests\kvcache_converter_test.py -q`. Current export
verification produced valid
converted ONNX files for both pinned checkpoints, and the focused suite passes
9 tests against the pinned ONNX Release wheel. The wheel also passed its
upstream focused suite (218 tests). Baseline CPU smoke runs on the installed
stock ORT 1.31.0 with pretrained weights produced finite logits of shape
`(1, 2, 151936)` for Qwen and `(1, 2, 262144)` for Gemma. The real-weight
Qwen and Gemma comparisons above establish tensor-cache parity on the pinned
native runtime build.

## Deliberate scope limits

This example does not implement continuous batching, prefix sharing,
speculative decoding, quantized/offloaded caches, or GPU cache kernels. The
demonstration compares one sequence at a time and uses CPU float32. It does not
modify Mobius's core model implementations.
