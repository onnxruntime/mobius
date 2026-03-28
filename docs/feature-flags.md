# Feature Flags

mobius exposes a small set of runtime feature flags that control experimental
or environment-specific behaviour. Flags live in `mobius._flags` and are
accessible through the public API.

## Available flags

| Flag | Environment variable | Default | Description |
|------|---------------------|---------|-------------|
| `suppress_dedup_warning` | `MOBIUS_SUPPRESS_DEDUP_WARNING` | `True` | Suppress "has no constant value" warnings from the initializer-deduplication pass. These are expected noise before weights are loaded. Set to `False` to surface all deduplication warnings. |

## Setting flags via environment variables

Environment variables are read when the global `flags` singleton is constructed
at import time. Set them before importing mobius (e.g., in a shell or `.env`):

```bash
# Disable warning suppression to see all deduplication warnings
export MOBIUS_SUPPRESS_DEDUP_WARNING=0

python -c "import mobius; mobius.build('Qwen/Qwen2.5-0.5B-Instruct')"
```

Accepted truthy values: `1`, `true`, `yes` (case-insensitive).
Accepted falsy values: `0`, `false`, `no` (case-insensitive).
Any other value falls back to the field default.

## Setting flags programmatically

Assign directly to the `flags` singleton at any point after import:

```python
import mobius

# Disable warning suppression
mobius.flags.suppress_dedup_warning = False

# Re-enable it
mobius.flags.suppress_dedup_warning = True
```

## Using `override_flags()` in tests

For tests that need a temporary flag value, use the `override_flags` context
manager. It restores the original values on exit, even if the test raises:

```python
import mobius
from mobius import override_flags

def test_build_with_warnings(tmp_path):
    with override_flags(suppress_dedup_warning=False):
        pkg = mobius.build("Qwen/Qwen2.5-0.5B-Instruct")
    # suppress_dedup_warning is restored here
```

`override_flags` raises `ValueError` for unknown flag names, catching typos
early:

```python
# Raises: ValueError: Unknown flag name(s): typo. Available flags: suppress_dedup_warning
with override_flags(typo=True):
    ...
```

> **Thread safety:** `override_flags` is not thread-safe — concurrent calls
> in different threads may interleave the save/restore cycle. For pytest,
> this is safe with `pytest -n auto` because xdist workers are separate
> processes with independent flag singletons.

## Listing all flags

```python
import mobius

print(mobius.list_flags())
# {'suppress_dedup_warning': True}
```

## Adding new flags

1. Add a field to the `Flags` dataclass in `src/mobius/_flags.py`:

   ```python
   @dataclasses.dataclass
   class Flags:
       suppress_dedup_warning: bool = dataclasses.field(
           default_factory=lambda: _env_bool("MOBIUS_SUPPRESS_DEDUP_WARNING", True)
       )
       """Suppress deduplication pass warnings (see above)."""

       my_new_flag: bool = dataclasses.field(
           default_factory=lambda: _env_bool("MOBIUS_MY_NEW_FLAG", False)
       )
       """Short description of what my_new_flag controls."""
   ```

2. Wire the flag into the code path it controls.

3. Add the flag to the table at the top of this page.

4. Add tests in `src/mobius/_flags_test.py` following the existing patterns.
