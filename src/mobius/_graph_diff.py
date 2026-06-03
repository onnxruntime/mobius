# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Architecture diff: canonical graph comparison for ONNX models.

Provides three public functions:

- :func:`canonicalize_graph` — convert an ``onnx_ir.Graph`` into a
  name-independent dictionary describing its structure.
- :func:`diff_graphs` — compare two canonical forms and return a list of
  change records.
- :func:`render_markdown` — render a set of per-model diffs as
  GitHub-flavored Markdown suitable for a PR comment.
"""

from __future__ import annotations

import difflib
from typing import Any

import onnx_ir as ir

# ── Status emoji legend ──────────────────────────────────────────────
_STATUS_NO_CHANGE = "⚪"
_STATUS_MINOR = "🔵"
_STATUS_MODERATE = "🟡"
_STATUS_MAJOR = "🔴"


# =====================================================================
# 1. canonicalize_graph
# =====================================================================


def _dim_to_str(dim: int | Any) -> str:
    """Convert a shape dimension to a comparable string."""
    if isinstance(dim, int):
        return str(dim)
    # SymbolicDim or other — use "?" to ignore concrete names
    return "?"


def _shape_signature(value: ir.Value) -> list[str]:
    """Return a list of dimension strings for a value's shape."""
    if value.shape is None:
        return []
    return [_dim_to_str(d) for d in value.shape]


def _dtype_str(value: ir.Value) -> str:
    """Return a comparable dtype string for a value."""
    if value.dtype is not None:
        return str(value.dtype)
    return "UNKNOWN"


# Sentinel keys marking a recursively-canonicalised subgraph payload inside an
# attribute's comparable value.  diff_graphs uses these to render subgraph
# deltas readably instead of dumping a raw nested canonical dict.
_SUBGRAPH_KEY = "__subgraph__"
_SUBGRAPHS_KEY = "__subgraphs__"


def _attr_to_comparable(attr: ir.Attr) -> Any:
    """Convert an attribute to a JSON-serialisable comparable value.

    Most attributes reduce to their scalar / list value.  GRAPH-typed
    attributes (``If``'s ``then_branch`` / ``else_branch``, ``Loop`` /
    ``Scan`` bodies) are *recursively canonicalised* so subgraph structure
    participates in the diff — without this, per-layer phase-split ``If``
    subgraphs are invisible to the architecture diff.  Remaining opaque
    types (TENSOR, SPARSE_TENSOR, TYPE_PROTO, …) are recorded as their type
    string to keep canonicalisation lightweight.
    """
    simple_types = {
        ir.AttributeType.FLOAT,
        ir.AttributeType.INT,
        ir.AttributeType.STRING,
        ir.AttributeType.FLOATS,
        ir.AttributeType.INTS,
        ir.AttributeType.STRINGS,
    }
    if attr.type in simple_types:
        v = attr.value
        # Tuples → lists for JSON compatibility
        if isinstance(v, tuple):
            return list(v)
        return v
    # Recurse into subgraphs so their node structure is compared, not collapsed.
    # Subgraphs reuse canonicalize_graph, so inner node/value names are ignored
    # the same way top-level ones are (see canonicalize_graph's name-independence).
    if attr.type == ir.AttributeType.GRAPH:
        return {_SUBGRAPH_KEY: canonicalize_graph(attr.value)}
    if attr.type == ir.AttributeType.GRAPHS:
        return {_SUBGRAPHS_KEY: [canonicalize_graph(g) for g in attr.value]}
    # For TENSOR, SPARSE_TENSOR, TYPE_PROTO, … just record the type.
    return f"<{attr.type.name}>"


def canonicalize_graph(graph: ir.Graph) -> dict:
    """Convert an ``ir.Graph`` into a name-independent canonical form.

    The result is a plain ``dict`` that captures the graph *structure*
    without depending on any node or value names.  Two graphs that are
    structurally identical but use different names will produce equal
    canonical forms.

    Keys in the returned dict:

    ``interface``
        ``{"inputs": [...], "outputs": [...]}`` with dtype and symbolic
        shape per port.
    ``initializers``
        List of ``{"dtype": ..., "shape": [...]}`` sorted by shape then
        dtype.
    ``nodes``
        List of per-node dicts in topological order containing
        ``op_type``, ``domain``, ``attributes``, ``input_ids``,
        ``num_outputs``.
    ``op_sequence``
        Flat list of ``op_type`` strings (convenient for difflib).
    """
    # -- interface ---------------------------------------------------------
    interface: dict[str, list[dict]] = {"inputs": [], "outputs": []}
    for v in graph.inputs:
        interface["inputs"].append({"dtype": _dtype_str(v), "shape": _shape_signature(v)})
    for v in graph.outputs:
        interface["outputs"].append({"dtype": _dtype_str(v), "shape": _shape_signature(v)})

    # -- initializers (sorted, name-free) ----------------------------------
    init_infos: list[dict] = []
    for v in graph.initializers.values():
        dtype = "UNKNOWN"
        shape: list[str] = []
        if v.dtype is not None:
            dtype = str(v.dtype)
        if v.shape is not None:
            shape = [_dim_to_str(d) for d in v.shape]
        elif v.const_value is not None:
            cv = v.const_value
            if cv.dtype is not None:
                dtype = str(cv.dtype)
            if cv.shape is not None:
                shape = [str(d) for d in cv.shape]
        init_infos.append({"dtype": dtype, "shape": shape})
    # Sort for deterministic comparison (independent of insertion order)
    init_infos.sort(key=lambda d: (d["dtype"], d["shape"]))

    # -- nodes (topological order) -----------------------------------------
    # Assign positional IDs to every *value* so that connectivity can be
    # expressed without names.
    value_id: dict[int, int] = {}  # id(Value) → positional int
    counter = 0
    for v in graph.inputs:
        value_id[id(v)] = counter
        counter += 1
    for v in graph.initializers.values():
        if id(v) not in value_id:
            value_id[id(v)] = counter
            counter += 1

    node_list: list[dict] = []
    op_sequence: list[str] = []
    for node in graph:
        # Record input connectivity via positional IDs
        input_ids: list[int | None] = []
        for inp in node.inputs:
            if inp is None:
                input_ids.append(None)
            else:
                vid = value_id.get(id(inp))
                if vid is None:
                    vid = counter
                    value_id[id(inp)] = counter
                    counter += 1
                input_ids.append(vid)

        # Record attributes
        attrs: dict[str, Any] = {}
        for attr_name, attr in node.attributes.items():
            if not attr.is_ref():
                attrs[attr_name] = _attr_to_comparable(attr)

        node_dict = {
            "op_type": node.op_type,
            "domain": node.domain,
            "attributes": attrs,
            "input_ids": input_ids,
            "num_outputs": len(list(node.outputs)),
        }
        node_list.append(node_dict)
        op_sequence.append(node.op_type)

        # Assign IDs to outputs
        for out in node.outputs:
            if out is not None and id(out) not in value_id:
                value_id[id(out)] = counter
                counter += 1

    return {
        "interface": interface,
        "initializers": init_infos,
        "nodes": node_list,
        "op_sequence": op_sequence,
    }


# =====================================================================
# 2. diff_graphs
# =====================================================================


def _describe_port_diff(base_port: dict, head_port: dict) -> str:
    """Describe what changed between two I/O port dicts (dtype, shape)."""
    parts: list[str] = []
    bd, hd = base_port.get("dtype", "?"), head_port.get("dtype", "?")
    if bd != hd:
        parts.append(f"dtype {bd} → {hd}")
    bs, hs = base_port.get("shape", []), head_port.get("shape", [])
    if bs != hs:
        parts.append(f"shape {bs} → {hs}")
    return "; ".join(parts) or "changed"


def _is_subgraph_payload(value: Any) -> bool:
    """True if *value* is a recursively-canonicalised subgraph payload."""
    return isinstance(value, dict) and (_SUBGRAPH_KEY in value or _SUBGRAPHS_KEY in value)


def _subgraph_list(value: Any) -> list[dict]:
    """Extract the list of subgraph canonical forms from an attr payload."""
    if isinstance(value, dict):
        if _SUBGRAPH_KEY in value:
            return [value[_SUBGRAPH_KEY]]
        if _SUBGRAPHS_KEY in value:
            return list(value[_SUBGRAPHS_KEY])
    return []


# Nested sub-change types that make a subgraph delta *structurally* significant
# (a node/branch was added/removed, rewired, or the subgraph interface moved),
# as opposed to a mere inner-attribute tweak.  Any of these promotes the
# containing attribute to a subgraph_structure_change → MODERATE.  Note this is
# uniformly MODERATE: unlike a *top-level* interface_change (MAJOR, an external
# model-contract break), a subgraph's interface is internal control-flow plumbing,
# so it stays MODERATE here — a deliberate asymmetry.
# ``subgraph_structure_change`` is included so structural significance
# *propagates* upward through nested subgraphs (e.g. an If inside an If).
_STRUCTURAL_SUB_TYPES = frozenset(
    {
        "added_node",
        "removed_node",
        "changed_connectivity",
        "interface_change",
        "subgraph_structure_change",
    }
)


def _describe_subgraph_attr_change(key: str, base_val: Any, head_val: Any) -> tuple[str, bool]:
    """Describe a GRAPH-typed attribute change, recursing into the subgraph(s).

    Returns ``(detail, structural)`` where *detail* is a readable summary
    that surfaces the nested diff (e.g. ``"then_branch: node[0] Concat:
    axis: 0 → 1"`` or ``"then_branch: + Mul; - Add"``) and *structural* is
    True when the nested delta adds/removes a node or branch, rewires
    connectivity, or changes the subgraph interface — i.e. a change that
    should outrank a pure inner-attribute tweak.
    """
    base_subs = _subgraph_list(base_val)
    head_subs = _subgraph_list(head_val)
    count = max(len(base_subs), len(head_subs))
    structural = False
    parts: list[str] = []
    for idx in range(count):
        bs = base_subs[idx] if idx < len(base_subs) else None
        hs = head_subs[idx] if idx < len(head_subs) else None
        # The attribute *key* already names a single subgraph (then_branch,
        # body, …); only disambiguate by index when there are several (GRAPHS).
        label = "" if count == 1 else f"subgraph[{idx}]"
        if bs is None or hs is None:
            # A whole branch/body was added or removed.
            structural = True
            verb = "added" if bs is None else "removed"
            parts.append(f"{label} {verb}".strip())
            continue
        sub_changes = diff_graphs(bs, hs)
        if not sub_changes:
            continue
        if {c["type"] for c in sub_changes} & _STRUCTURAL_SUB_TYPES:
            structural = True
        inner = "; ".join(c["details"] for c in sub_changes)
        parts.append(f"{label}: {inner}" if label else inner)
    detail = f"{key}: " + ("; ".join(parts) if parts else "subgraph changed")
    return detail, structural


def diff_graphs(base: dict, head: dict) -> list[dict[str, Any]]:
    """Compare two canonical graph representations.

    Returns a list of change dicts.  Each dict has a ``"type"`` key with
    one of: ``"added_node"``, ``"removed_node"``, ``"changed_attrs"``,
    ``"subgraph_structure_change"``, ``"changed_connectivity"``,
    ``"interface_change"``, ``"initializer_change"``.  A ``"details"``
    key carries human-readable information about the change.

    ``subgraph_structure_change`` is emitted for a GRAPH-typed attribute
    (``If`` branches, ``Loop`` / ``Scan`` bodies) whose nested graph gains
    or loses a node/branch, is rewired, or changes interface; a subgraph
    delta that only tweaks an inner attribute stays ``changed_attrs``.
    """
    changes: list[dict[str, Any]] = []

    # -- interface diff ----------------------------------------------------
    base_iface = base.get("interface", {})
    head_iface = head.get("interface", {})
    if base_iface != head_iface:
        details_parts: list[str] = []
        b_in = base_iface.get("inputs", [])
        h_in = head_iface.get("inputs", [])
        if len(b_in) != len(h_in):
            details_parts.append(f"input count {len(b_in)} → {len(h_in)}")
        b_out = base_iface.get("outputs", [])
        h_out = head_iface.get("outputs", [])
        if len(b_out) != len(h_out):
            details_parts.append(f"output count {len(b_out)} → {len(h_out)}")
        # Check for dtype/shape changes on matched ports
        for idx, (bv, hv) in enumerate(zip(b_in, h_in)):
            if bv != hv:
                diffs = _describe_port_diff(bv, hv)
                details_parts.append(f"input[{idx}]: {diffs}")
        for idx, (bv, hv) in enumerate(zip(b_out, h_out)):
            if bv != hv:
                diffs = _describe_port_diff(bv, hv)
                details_parts.append(f"output[{idx}]: {diffs}")
        changes.append(
            {
                "type": "interface_change",
                "details": "; ".join(details_parts) or "interface changed",
            }
        )

    # -- initializer diff --------------------------------------------------
    b_inits = base.get("initializers", [])
    h_inits = head.get("initializers", [])
    if b_inits != h_inits:
        details_parts_init: list[str] = []
        if len(b_inits) != len(h_inits):
            details_parts_init.append(f"count {len(b_inits)} → {len(h_inits)}")
        # Summarize dtype distribution changes
        from collections import Counter

        b_dtypes = Counter(i["dtype"] for i in b_inits)
        h_dtypes = Counter(i["dtype"] for i in h_inits)
        if b_dtypes != h_dtypes:
            dtype_parts = []
            for dt in sorted(set(b_dtypes) | set(h_dtypes)):
                bc, hc = b_dtypes.get(dt, 0), h_dtypes.get(dt, 0)
                if bc != hc:
                    dtype_parts.append(f"{dt}: {bc} → {hc}")
            if dtype_parts:
                details_parts_init.append("dtype distribution: " + ", ".join(dtype_parts))
        changes.append(
            {
                "type": "initializer_change",
                "details": "; ".join(details_parts_init) or "initializers changed",
            }
        )

    # -- op-sequence diff (added / removed nodes) --------------------------
    base_ops = base.get("op_sequence", [])
    head_ops = head.get("op_sequence", [])

    diff_lines = list(
        difflib.unified_diff(
            base_ops,
            head_ops,
            fromfile="base",
            tofile="head",
            lineterm="",
        )
    )

    for line in diff_lines:
        if line.startswith("+") and not line.startswith("+++"):
            op = line[1:]
            changes.append({"type": "added_node", "details": f"+ {op}"})
        elif line.startswith("-") and not line.startswith("---"):
            op = line[1:]
            changes.append({"type": "removed_node", "details": f"- {op}"})

    # -- attribute diff on positionally matched nodes ----------------------
    base_nodes = base.get("nodes", [])
    head_nodes = head.get("nodes", [])
    limit = min(len(base_nodes), len(head_nodes))
    for i in range(limit):
        bn = base_nodes[i]
        hn = head_nodes[i]
        if bn["op_type"] != hn["op_type"]:
            # Already captured by the op-sequence diff
            continue
        if bn["attributes"] != hn["attributes"]:
            ba = bn["attributes"]
            ha = hn["attributes"]
            plain_details: list[str] = []
            all_keys = sorted(set(ba) | set(ha))
            for k in all_keys:
                bv = ba.get(k)
                hv = ha.get(k)
                if bv == hv:
                    continue
                if _is_subgraph_payload(bv) or _is_subgraph_payload(hv):
                    detail, structural = _describe_subgraph_attr_change(k, bv, hv)
                    changes.append(
                        {
                            "type": (
                                "subgraph_structure_change" if structural else "changed_attrs"
                            ),
                            "details": f"node[{i}] {bn['op_type']}: {detail}",
                        }
                    )
                else:
                    plain_details.append(f"{k}: {bv!r} → {hv!r}")
            if plain_details:
                changes.append(
                    {
                        "type": "changed_attrs",
                        "details": (f"node[{i}] {bn['op_type']}: " + ", ".join(plain_details)),
                    }
                )
        if bn["input_ids"] != hn["input_ids"]:
            changes.append(
                {
                    "type": "changed_connectivity",
                    "details": (
                        f"node[{i}] {bn['op_type']}: "
                        f"input_ids {bn['input_ids']} → {hn['input_ids']}"
                    ),
                }
            )

    return changes


# =====================================================================
# 3. render_markdown
# =====================================================================


def _change_status(change_list: list[dict[str, Any]]) -> str:
    """Pick a status emoji based on the severity of changes."""
    if not change_list:
        return _STATUS_NO_CHANGE
    types = {c["type"] for c in change_list}
    if types & {"interface_change"}:
        return _STATUS_MAJOR
    if types & {
        "added_node",
        "removed_node",
        "changed_connectivity",
        "subgraph_structure_change",
    }:
        return _STATUS_MODERATE
    if types & {"changed_attrs", "initializer_change"}:
        return _STATUS_MINOR
    return _STATUS_MINOR


def _op_diff_block(base_ops: list[str], head_ops: list[str]) -> str:
    """Return a fenced ``diff`` code block from two op-type sequences."""
    lines = list(
        difflib.unified_diff(
            base_ops,
            head_ops,
            fromfile="base",
            tofile="head",
            lineterm="",
        )
    )
    if not lines:
        return "_No op-sequence changes._"
    return "```diff\n" + "\n".join(lines) + "\n```"


def render_markdown(
    diffs: dict[str, dict[str, dict[str, Any]]],
    *,
    base_ref: str | None = None,
    head_ref: str | None = None,
    repo_url: str | None = None,
) -> str:
    """Render diffs for all affected models as GitHub-flavored Markdown.

    Args:
        diffs: Mapping of ``model_type`` → ``{sub_model_name: [changes]}``
            where each change comes from :func:`diff_graphs`.  The inner
            dict also carries ``"_base_ops"`` and ``"_head_ops"`` lists
            for rendering the diff block, and ``"_base_node_count"`` /
            ``"_head_node_count"`` ints for the summary.
        base_ref: Short SHA or ref name for the base commit.
        head_ref: Short SHA or ref name for the head commit.
        repo_url: GitHub repository URL (e.g. ``https://github.com/org/repo``).
            When provided together with *base_ref* and *head_ref*, each SHA is
            rendered as a clickable link to the commit page.

    Returns:
        A Markdown string.
    """
    lines: list[str] = []
    lines.append("<!-- arch-diff-bot -->")
    lines.append("## 🏗️ Architecture Diff\n")

    if base_ref and head_ref:

        def _sha_link(sha: str) -> str:
            if repo_url:
                return f"[`{sha}`]({repo_url}/commit/{sha})"
            return f"`{sha}`"

        lines.append(f"Comparing {_sha_link(base_ref)} → {_sha_link(head_ref)}\n")

    # ── summary table ────────────────────────────────────────────────
    lines.append("| Model | Sub-model | Changes | Status |")
    lines.append("|-------|-----------|---------|--------|")

    any_change = False
    for model_type in sorted(diffs):
        sub_models = diffs[model_type]
        for sub_name in sorted(sub_models):
            entry = sub_models[sub_name]
            change_list: list[dict[str, Any]] = entry.get("changes", [])
            status = _change_status(change_list)
            n_changes = len(change_list)
            if n_changes > 0:
                any_change = True
            lines.append(f"| {model_type} | {sub_name} | {n_changes} | {status} |")

    if not any_change:
        lines.append("")
        lines.append("**No architecture changes detected.** ✅")
        lines.append("")
        _append_legend(lines)
        return "\n".join(lines)

    lines.append("")

    # ── details per model ────────────────────────────────────────────
    for model_type in sorted(diffs):
        sub_models = diffs[model_type]
        for sub_name in sorted(sub_models):
            entry = sub_models[sub_name]
            change_list = entry.get("changes", [])
            if not change_list:
                continue

            b_count = entry.get("_base_node_count", 0)
            h_count = entry.get("_head_node_count", 0)
            base_ops = entry.get("_base_ops", [])
            head_ops = entry.get("_head_ops", [])

            lines.append(
                f"<details><summary><b>{model_type} / "
                f"{sub_name}</b> — {len(change_list)} "
                f"change(s)</summary>\n"
            )

            lines.append(f"**Op summary:** {b_count} → {h_count} nodes\n")

            # Op diff block
            lines.append(_op_diff_block(base_ops, head_ops))
            lines.append("")

            # Group changes by type
            added = [c for c in change_list if c["type"] == "added_node"]
            removed = [c for c in change_list if c["type"] == "removed_node"]
            attrs = [c for c in change_list if c["type"] == "changed_attrs"]
            connectivity = [c for c in change_list if c["type"] == "changed_connectivity"]
            subgraph = [c for c in change_list if c["type"] == "subgraph_structure_change"]
            iface = [c for c in change_list if c["type"] == "interface_change"]
            inits = [c for c in change_list if c["type"] == "initializer_change"]

            if added:
                lines.append("**Added nodes:**")
                for c in added:
                    lines.append(f"- `{c['details']}`")
                lines.append("")

            if removed:
                lines.append("**Removed nodes:**")
                for c in removed:
                    lines.append(f"- `{c['details']}`")
                lines.append("")

            if subgraph:
                lines.append("**Subgraph structure changes:**")
                for c in subgraph:
                    lines.append(f"- `{c['details']}`")
                lines.append("")

            if attrs:
                lines.append("**Modified attributes:**")
                for c in attrs:
                    lines.append(f"- `{c['details']}`")
                lines.append("")

            if connectivity:
                lines.append("**Connectivity changes:**")
                for c in connectivity:
                    lines.append(f"- `{c['details']}`")
                lines.append("")

            if iface:
                lines.append("**Interface changes:**")
                for c in iface:
                    lines.append(f"- {c['details']}")
                lines.append("")

            if inits:
                lines.append("**Initializer changes:**")
                for c in inits:
                    lines.append(f"- {c['details']}")
                lines.append("")

            lines.append("</details>\n")

    _append_legend(lines)
    return "\n".join(lines)


def _append_legend(lines: list[str]) -> None:
    """Append the emoji legend to *lines*."""
    lines.append("---")
    lines.append(
        f"**Legend:** {_STATUS_NO_CHANGE} No change · "
        f"{_STATUS_MINOR} Minor (attrs/inits) · "
        f"{_STATUS_MODERATE} Moderate (nodes added/removed) · "
        f"{_STATUS_MAJOR} Major (interface changed)"
    )
