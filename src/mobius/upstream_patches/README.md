# upstream_patches

Corrections to runtime assets that ship broken from upstream model repos.

Weights are not the whole package. The chat template, tokenizer config and
processor config decide whether an agent client can actually drive the model,
and when one of them is wrong the export is faithful and the deployment is
still broken. Fixing it by hand does not stick: the defect returns on the next
re-export, and on every other machine.

## Layout

```
data/<owner>/<model>/
    patch.json       what to change, and which upstream revision it was written against
    README.md        the symptom, the defect, and why this is the right fix
    <target>.patch   a unified diff
```

## How a correction finds its target

By the sha256 of the file already in the package, not by a model id. A source
can be a Hub repo, a snapshot directory or a local checkout, and the hash is
the same in all three. It also means a correction can only ever rewrite bytes
it recognises.

When upstream republishes the file, the hash stops matching and the correction
stops being applied. That is the intended behaviour: a stale correction does
nothing rather than corrupting a package. `check_upstream` is how you find out.

```
python -m mobius.upstream_patches.check_upstream
python -m mobius.upstream_patches.check_upstream meta-models/Muse-Glimmer-30B
```

It exits non-zero when a recorded file changed upstream. That is not a failure,
it is a review request: re-verify, then either refresh `revision` and
`upstream_sha256`, or delete the correction because upstream fixed it.

## Operations

`diff` applies a unified diff to one file.

```json
{
  "kind": "diff",
  "target": "chat_template.jinja",
  "patch": "chat_template.jinja.patch",
  "upstream_sha256": "<sha256 of the upstream file this was written against>"
}
```

Diffs are applied strictly: the hunk lands at the line it names with the
context it names, or it does not land. No fuzz, no offset search.

`sync_chat_template` rewrites the `chat_template` key of a JSON config from an
already-corrected template. Templates get duplicated into
`tokenizer_config.json`, and a package whose two copies disagree fails in a way
that depends on which loader you use. It runs only if the template it mirrors
was actually corrected, so it cannot touch a package the diff skipped.

Prefer a narrow operation over replacing a file wholesale. Checking in a whole
`tokenizer_config.json` would pin every unrelated key in it to whatever
upstream happened to have that day.

## Adding a correction

1. Confirm the defect is upstream and not in the exporter. Diff the file in the
   HF snapshot against what you believe it should be.
2. Write the diff against the snapshot, and record its revision and sha256.
3. Explain the defect in `README.md` in terms of the symptom a user sees.
4. Verify end to end against a real client, and say so.

> [!NOTE]
> Preference goes to upstream. Open an issue or a PR there first and keep the
> correction only for as long as it takes to land.
