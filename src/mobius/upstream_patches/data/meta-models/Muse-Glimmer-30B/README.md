# meta-models/Muse-Glimmer-30B

## The defect

`chat_template.jinja` derives a namespace from every offered tool name and
renders it as a wildcard pattern:

```jinja
{%- set tns = fn.name.split('.')[0] -%}
...
{%- set rns.recipients = rns.recipients + ['"' + tns + '.*"'] -%}
```

`split('.')[0]` on an unqualified name returns the name itself, so a client
offering plain tool names gets a system preamble that reads:

```
# Valid recipients: "self", "read.*", "bash.*", "glob.*", "user".
```

Nothing named `read` is addressable any more -- only things *under* `read`.
The model does what the preamble asks and supplies a second segment,
usually by borrowing a parameter name, and emits a call to `read.filePath`.
The harness rejects it because no such tool was offered.

Observed against every OpenCode session: the first tool call of each task
failed, the agent recovered on retry, and every task paid for the round trip.

## The fix

Treat the dot as the signal it is. A name that contains one is namespaced and
keeps the `"ns.*"` pattern; a name without one is already the whole address
and is rendered verbatim.

The same guard is applied to the tool metadata block, which otherwise emits a
namespace entry for a tool that has no namespace.

## Scope

Clients that offer namespaced tool names are unaffected -- they render exactly
as before. The change only reaches names that upstream was mangling.

## Verification

Rendered both shapes with jinja2 and diffed the preamble; ran a read task and
a multi-tool debug task (read, edit, bash, todowrite) end to end through
OpenCode against the INT4 CUDA export with no invalid tool errors.
