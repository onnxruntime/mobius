# Vendored onnx-genai metadata schema

`inference_metadata.schema.json` is a verbatim copy of onnx-genai's published
`schema/inference_metadata.schema.json`, which that repo generates from its Rust
types (`crates/onnx-genai-metadata/src/schema/`).

**Why it is vendored.** The schema conformance tests used to look for a local
onnx-genai checkout and `pytest.skip` when they could not find one. CI has no
such checkout, so every one of those tests skipped there and the emitters were
only ever validated on the few developer machines that happened to have
onnx-genai cloned — and then only against whatever revision that clone sat on.
Two upstream contract redesigns (`pipeline` becoming a `PipelineSpec` whose only
property is `workflow`, and `speculative` becoming a `SpeculativeContract`) went
unnoticed for exactly that reason. Pinning the schema here makes the contract
mobius targets an explicit, reviewable file and makes drift a CI failure rather
than a silent skip.

**Updating it.** Copy the file from onnx-genai `main` and run the onnx-genai
tests in this package:

```bash
cp <onnx-genai>/schema/inference_metadata.schema.json \
   src/mobius/integrations/onnx_genai/_schema/
python -m pytest src/mobius/integrations/onnx_genai/ -q
```

Set `ONNX_GENAI_SCHEMA=/path/to/inference_metadata.schema.json` to validate
against a different revision without editing this copy.

Synced from onnx-genai `cb7baf924` (2026-08-25).
