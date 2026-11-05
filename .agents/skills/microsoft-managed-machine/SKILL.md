---
name: microsoft-managed-machine
description: Applies on Microsoft-managed machines when installing Python/PyPI packages and when the PyPI feed is unreachable.
---

# Microsoft-managed machine

Install Python packages through the Microsoft package feed proxy:

```bash
pip install <package> --index-url https://packagefeedproxy.microsoft.io/pypi/simple
```

**Important:** This feed is NOT designed for CIs. DO NOT use it on non-Microsoft-managed machines.
