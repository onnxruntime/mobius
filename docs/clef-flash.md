# Clef-Flash decision models

Mobius supports [`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash):
a Qwen3.5 multimodal backbone plus a separately published joint schema head.
Unlike a language model, it scores the allowed answers to all schema fields in
one full-record forward pass. Boolean (`noul`), categorical (`choice`), and
ordinal (`score`) questions return labeled probabilities, not generated text.

## Export

```python
import mobius

package = mobius.build("Cloudflare/clef-flash", dtype="f32")
package.save("clef_onnx", max_workers=1)
```

The official checkpoint defaults to revision
`fde727a287004204b7518dcc983fe64379776712`. An explicit `revision=` overrides it
for both backbone and head assets. Local checkpoint directories must contain
`config.json`, backbone safetensors, `joint_head_config.json`, and
`joint_head.safetensors`. Mobius does not execute the checkpoint's remote Python.
Weights are loaded lazily, including splitting the head's fused attention QKV
tensors; serialization streams them rather than materializing the full checkpoint.
The 9B backbone still requires substantial disk space and inference memory.

The CLI also exports all four ONNX components:

```powershell
mobius build --model Cloudflare/clef-flash --dtype f32 clef_onnx
```

The four saved components have the following contracts:

| Component | Inputs | Outputs |
|---|---|---|
| `vision_encoder` | Packed `pixel_values`, `image_grid_thw` | `image_features` |
| `embedding` | `input_ids`, separate `image_features` and `video_features` | `inputs_embeds` |
| `decoder` | `inputs_embeds [1,S,H]`, `attention_mask [1,S]`, `position_ids [3,1,S]` | `hidden_states [1,S,H]` |
| `decision_head` | Hidden states, input IDs, schema spans, option owners, question types | Flat `logits [O]`, grouped `probabilities [O]` |

Run each record separately without padding and use an all-ones attention mask.
The last prompt token is the global evidence anchor. Internal convolution and
recurrent states start at zero; there is no exposed autoregressive cache.
DeltaNet recurrence and schema prefix pooling accumulate in float32 even when
weights and hidden states use reduced precision.
Question and option spans are nonempty, half-open intervals into the full prompt.
The head uses the **output LM embedding table** for lexical features, which is
not interchangeable with the input embedding table.

## Encoding and inference

`mobius.integrations.clef.encode_clef_record` reproduces the publisher's prompt
and span protocol. It truncates only state text, preserving schema and media.
`ClefRecord.decision_feeds(hidden_states)` prepares the head inputs, and
`ClefRecord.probabilities_by_question(probabilities)` restores the original labels.

For example, a record may contain:

```json
{
  "state": {"message": "An overdue invoice"},
  "questions": {
    "urgent": {"type": "noul", "instructions": "Does this require attention?"},
    "team": {
      "type": "choice",
      "criteria": {"billing": "Billing", "support": "Support"}
    },
    "priority": {"type": "score", "criteria": ["low", "high"]}
  }
}
```

Save this JSON as `record.json`, then run the text-only example:

```powershell
python examples\clef_flash.py --record record.json --output clef_onnx
```

For media, pass the pinned Hugging Face processor to `encode_clef_record`.
Its packed pixel rows, image/video grids, and multimedia token metadata are
returned in `record.media`. Run the vision encoder separately for images and
videos (feed video grids through its `image_grid_thw` port), then provide the
respective features to the embedding mixer; absent streams use `[0,H]` arrays.
Compute Qwen3.5's three-axis MRoPE positions from the complete expanded prompt and
the processor grids/token metadata. Text-only positions are three identical
copies of `arange(S)`; that shortcut is **not correct for media**.

## Validation and limitations

Dedicated tests cover graph construction in float32/float16/bfloat16,
schema encoding against the pinned publisher implementation, float32/float16
decision-head numerical parity, and streamed tiny-checkpoint float32/float16 parity
through the complete text/image/video/mixed-media pipeline. The latter covers
both hybrid and full-attention backbones, including conflicting legacy/nested
rotary settings. Tests use randomly initialized, reduced-size backbones and the
publisher's head code, not the downloaded 9B backbone.

Full real-checkpoint inference and decision goldens have not been established.
BF16 numerical parity needs CUDA kernels and is skipped on CPU; full-pipeline
bfloat16 and CUDA/DML parity are not certified. Quantized source
checkpoints, DeepStack, padding/batching, and autoregressive export options are
unsupported. Olive quantization has not been validated for this package.

The generic causal-LM golden/generation harness does not implement this decision
contract, so its L4/L5 generation tests are not applicable. Use the dedicated
Clef tests rather than treating successful generic Qwen3.5 generation as evidence
for the joint head. ORT GenAI configuration is explicitly rejected; onnx-genai
export writes advisory component contracts only. Use ONNX Runtime with host-owned
orchestration, as shown in the example.
