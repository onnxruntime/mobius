# EP-Aware Building: Quick Start

Pass `execution_provider` to select Mobius graph-construction and runtime
packaging contracts:

```python
import mobius

# Portable construction path
package = mobius.build("meta-llama/Llama-3.2-1B")

# CUDA construction and runtime contract
package = mobius.build(
    "meta-llama/Llama-3.2-1B",
    execution_provider="cuda",
    dtype="f16",
)

package.save("output/llama/")
```

CLI equivalent:

```bash
mobius build --model meta-llama/Llama-3.2-1B \
  --ep cuda --dtype f16 output/llama/
```

Mobius performs exporter cleanup and local-function inlining, but it does not
apply post-export graph fusions or EP-specific lowerings. Run the appropriate
Olive optimization workflow on the exported package before deployment.

Use `mobius list eps` or the Python registry to inspect available providers:

```python
from mobius import ep_registry, get_ep

print(sorted(ep_registry))
print(get_ep("cuda").provider_options)
```

See [Execution Provider Aware Building](execution_providers.md) for the full
construction and packaging contract.
