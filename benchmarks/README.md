# Muse workflow benchmark

`muse_workflow_h200.json` is the shared workload for a paired native and
metadata-workflow benchmark of the published Muse Glimmer INT4 package. Both
paths use the exact 68 token IDs in `muse_prompt_ids.json`, greedy parameters,
token budget, warmups, and steady-decode window. Passing token IDs avoids a
tokenizer adding another beginning-of-text token to the already rendered prompt.
Both runners set `ORT_ENABLE_CUDNN_FLASH_ATTENTION=0`; this matches the native
baseline and avoids comparing different GQA attention kernels.
`request_max_length` is prompt tokens plus new tokens (68 + 128 = 196);
`model_max_context` is the independent 131072-token artifact admission ceiling.
For the text-only VLM path, omit `request.image`. The runtime initializes
`request.image_present=false`, and the false branch supplies empty image features.

Run the native ORT GenAI path:

```bash
python scripts/benchmark_muse_native.py \
  --model artifacts/muse-int4-package \
  --output artifacts/muse-int4-package/native-benchmark.json
```

Run the ONNX GenAI workflow path with the `profile_native` binary built from
the schema/runtime revision named in the JSON config:

```bash
python scripts/benchmark_muse_workflow.py \
  --model artifacts/muse-int4-package \
  --runner path/to/profile_native \
  --output artifacts/muse-int4-package/workflow-benchmark.json
```

The workflow runner must support `--pipeline --backend ort --ep cuda`,
`--prompt-ids`, and an optional `--image` request binding. The text-only workload
intentionally leaves the image unset so results remain comparable to the
published 61.76 tok/s baseline.
