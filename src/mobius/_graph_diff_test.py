# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for the _graph_diff module.

Tests the three public functions: ``canonicalize_graph``,
``diff_graphs``, and ``render_markdown``.
"""

from __future__ import annotations

import onnx_ir as ir

from mobius._graph_diff import (
    _SUBGRAPH_KEY,
    _SUBGRAPHS_KEY,
    canonicalize_graph,
    diff_graphs,
    render_markdown,
)

# ------------------------------------------------------------------
# Helpers — build tiny test graphs using onnx_ir directly
# ------------------------------------------------------------------


def _simple_add_graph(
    *,
    input_name: str = "x",
    bias_name: str = "bias",
    output_name: str = "y",
    node_name: str = "add_0",
) -> ir.Graph:
    """Return a graph: y = x + bias."""
    import numpy as np

    x = ir.val(
        input_name,
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape([1, 4]),
    )
    bias_tensor = ir.Tensor(np.ones(4, dtype=np.float32), name=bias_name)
    bias = ir.Value(
        name=bias_name,
        const_value=bias_tensor,
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape([4]),
    )
    add_node = ir.Node("", "Add", [x, bias], name=node_name)
    out = add_node.outputs[0]
    out.name = output_name
    return ir.Graph(
        [x],
        [out],
        nodes=[add_node],
        initializers=[bias],
    )


def _add_relu_graph(
    *,
    input_name: str = "a",
    bias_name: str = "b",
    mid_name: str = "mid",
    output_name: str = "z",
) -> ir.Graph:
    """Return a graph: z = Relu(a + b)."""
    import numpy as np

    x = ir.val(
        input_name,
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape([1, 4]),
    )
    bias_tensor = ir.Tensor(np.ones(4, dtype=np.float32), name=bias_name)
    bias = ir.Value(
        name=bias_name,
        const_value=bias_tensor,
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape([4]),
    )
    add_node = ir.Node("", "Add", [x, bias], name="add")
    add_out = add_node.outputs[0]
    add_out.name = mid_name

    relu_node = ir.Node("", "Relu", [add_out], name="relu")
    relu_out = relu_node.outputs[0]
    relu_out.name = output_name

    return ir.Graph(
        [x],
        [relu_out],
        nodes=[add_node, relu_node],
        initializers=[bias],
    )


# ------------------------------------------------------------------
# canonicalize_graph tests
# ------------------------------------------------------------------


class TestCanonicalizeGraph:
    """Tests for canonicalize_graph."""

    def test_empty_graph(self) -> None:
        graph = ir.Graph([], [], nodes=[])
        canon = canonicalize_graph(graph)

        assert canon["interface"] == {
            "inputs": [],
            "outputs": [],
        }
        assert canon["initializers"] == []
        assert canon["nodes"] == []
        assert canon["op_sequence"] == []

    def test_ignores_names(self) -> None:
        """Two structurally identical graphs with different names."""
        g1 = _simple_add_graph(
            input_name="input_0",
            bias_name="weight_0",
            output_name="result",
            node_name="node_add",
        )
        g2 = _simple_add_graph(
            input_name="x",
            bias_name="b",
            output_name="y",
            node_name="my_add",
        )
        c1 = canonicalize_graph(g1)
        c2 = canonicalize_graph(g2)

        assert c1 == c2

    def test_captures_op_sequence(self) -> None:
        g = _add_relu_graph()
        canon = canonicalize_graph(g)
        assert canon["op_sequence"] == ["Add", "Relu"]

    def test_captures_interface(self) -> None:
        g = _simple_add_graph()
        canon = canonicalize_graph(g)
        iface = canon["interface"]
        assert len(iface["inputs"]) == 1
        assert iface["inputs"][0]["dtype"] == "FLOAT"
        assert len(iface["outputs"]) == 1

    def test_captures_initializers(self) -> None:
        g = _simple_add_graph()
        canon = canonicalize_graph(g)
        assert len(canon["initializers"]) == 1
        init = canon["initializers"][0]
        assert init["dtype"] == "FLOAT"
        assert init["shape"] == ["4"]

    def test_captures_attributes(self) -> None:
        x = ir.val(
            "x",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([2, 3]),
        )
        concat = ir.Node(
            "",
            "Concat",
            [x, x],
            [ir.AttrInt64("axis", 1)],
            name="concat",
        )
        out = concat.outputs[0]
        out.name = "out"
        g = ir.Graph([x], [out], nodes=[concat])
        canon = canonicalize_graph(g)
        assert canon["nodes"][0]["attributes"] == {"axis": 1}

    def test_symbolic_dims_erased(self) -> None:
        """Differently-named symbolic dims produce equal canonical forms."""
        x1 = ir.val(
            "x",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(["batch", 4]),
        )
        add1 = ir.Node("", "Add", [x1, x1], name="add")
        o1 = add1.outputs[0]
        o1.name = "y"
        g1 = ir.Graph([x1], [o1], nodes=[add1])

        x2 = ir.val(
            "x",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(["B", 4]),
        )
        add2 = ir.Node("", "Add", [x2, x2], name="add")
        o2 = add2.outputs[0]
        o2.name = "y"
        g2 = ir.Graph([x2], [o2], nodes=[add2])

        assert canonicalize_graph(g1) == canonicalize_graph(g2)

    def test_captures_num_outputs(self) -> None:
        x = ir.val(
            "x",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([2, 3]),
        )
        split = ir.Node(
            "",
            "Split",
            [x],
            [ir.AttrInt64("num_outputs", 2)],
            num_outputs=2,
            name="split",
        )
        o1 = split.outputs[0]
        o1.name = "o1"
        o2 = split.outputs[1]
        o2.name = "o2"
        g = ir.Graph([x], [o1, o2], nodes=[split])
        canon = canonicalize_graph(g)
        assert canon["nodes"][0]["num_outputs"] == 2


# ------------------------------------------------------------------
# diff_graphs tests
# ------------------------------------------------------------------


class TestDiffGraphs:
    """Tests for diff_graphs."""

    def test_no_changes(self) -> None:
        g = _simple_add_graph()
        c = canonicalize_graph(g)
        changes = diff_graphs(c, c)
        assert changes == []

    def test_added_node(self) -> None:
        base = _simple_add_graph()
        head = _add_relu_graph()
        cb = canonicalize_graph(base)
        ch = canonicalize_graph(head)
        changes = diff_graphs(cb, ch)
        types = {c["type"] for c in changes}
        assert "added_node" in types
        added = [c for c in changes if c["type"] == "added_node"]
        assert any("Relu" in c["details"] for c in added)

    def test_removed_node(self) -> None:
        base = _add_relu_graph()
        head = _simple_add_graph()
        cb = canonicalize_graph(base)
        ch = canonicalize_graph(head)
        changes = diff_graphs(cb, ch)
        types = {c["type"] for c in changes}
        assert "removed_node" in types
        removed = [c for c in changes if c["type"] == "removed_node"]
        assert any("Relu" in c["details"] for c in removed)

    def test_changed_attributes(self) -> None:
        x = ir.val(
            "x",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([2, 3]),
        )
        n1 = ir.Node(
            "",
            "Concat",
            [x, x],
            [ir.AttrInt64("axis", 0)],
            name="c",
        )
        o1 = n1.outputs[0]
        o1.name = "out"
        g1 = ir.Graph([x], [o1], nodes=[n1])

        x2 = ir.val(
            "x",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([2, 3]),
        )
        n2 = ir.Node(
            "",
            "Concat",
            [x2, x2],
            [ir.AttrInt64("axis", 1)],
            name="c2",
        )
        o2 = n2.outputs[0]
        o2.name = "out2"
        g2 = ir.Graph([x2], [o2], nodes=[n2])

        changes = diff_graphs(canonicalize_graph(g1), canonicalize_graph(g2))
        types = {c["type"] for c in changes}
        assert "changed_attrs" in types
        attr_changes = [c for c in changes if c["type"] == "changed_attrs"]
        assert any("axis" in c["details"] for c in attr_changes)

    def test_interface_change(self) -> None:
        """Detect when an output is added."""
        import numpy as np

        x = ir.val(
            "x",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([1, 4]),
        )
        bias_t = ir.Tensor(np.ones(4, dtype=np.float32), name="b")
        bias = ir.Value(
            name="b",
            const_value=bias_t,
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([4]),
        )
        add = ir.Node("", "Add", [x, bias], name="add")
        out = add.outputs[0]
        out.name = "y"
        g1 = ir.Graph([x], [out], nodes=[add], initializers=[bias])

        # Build g2 with two outputs (identity branch)
        x2 = ir.val(
            "x",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([1, 4]),
        )
        bias_t2 = ir.Tensor(np.ones(4, dtype=np.float32), name="b")
        bias2 = ir.Value(
            name="b",
            const_value=bias_t2,
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([4]),
        )
        add2 = ir.Node("", "Add", [x2, bias2], name="add")
        out2 = add2.outputs[0]
        out2.name = "y"
        ident = ir.Node("", "Identity", [out2], name="ident")
        out3 = ident.outputs[0]
        out3.name = "y2"
        g2 = ir.Graph(
            [x2],
            [out2, out3],
            nodes=[add2, ident],
            initializers=[bias2],
        )

        changes = diff_graphs(canonicalize_graph(g1), canonicalize_graph(g2))
        types = {c["type"] for c in changes}
        assert "interface_change" in types

    def test_changed_connectivity(self) -> None:
        """Detect when node input wiring changes."""
        import numpy as np

        # Graph 1: Add(x, bias)
        x1 = ir.val("x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 4]))
        bias_t1 = ir.Tensor(np.ones(4, dtype=np.float32), name="b")
        bias1 = ir.Value(
            name="b",
            const_value=bias_t1,
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([4]),
        )
        add1 = ir.Node("", "Add", [x1, bias1], name="add")
        out1 = add1.outputs[0]
        out1.name = "y"
        g1 = ir.Graph([x1], [out1], nodes=[add1], initializers=[bias1])

        # Graph 2: Add(x, x) — same op, different wiring
        x2 = ir.val("x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 4]))
        bias_t2 = ir.Tensor(np.ones(4, dtype=np.float32), name="b")
        bias2 = ir.Value(
            name="b",
            const_value=bias_t2,
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([4]),
        )
        add2 = ir.Node("", "Add", [x2, x2], name="add")
        out2 = add2.outputs[0]
        out2.name = "y"
        g2 = ir.Graph([x2], [out2], nodes=[add2], initializers=[bias2])

        changes = diff_graphs(canonicalize_graph(g1), canonicalize_graph(g2))
        types = {c["type"] for c in changes}
        assert "changed_connectivity" in types
        conn = [c for c in changes if c["type"] == "changed_connectivity"]
        assert any("input_ids" in c["details"] for c in conn)

    def test_changed_connectivity_swapped_inputs(self) -> None:
        """Detect when two inputs to the same node are swapped."""
        import numpy as np

        # Graph 1: Sub(x, bias)
        x1 = ir.val("x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 4]))
        bias_t1 = ir.Tensor(np.ones(4, dtype=np.float32), name="b")
        bias1 = ir.Value(
            name="b",
            const_value=bias_t1,
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([4]),
        )
        sub1 = ir.Node("", "Sub", [x1, bias1], name="sub")
        out1 = sub1.outputs[0]
        out1.name = "y"
        g1 = ir.Graph([x1], [out1], nodes=[sub1], initializers=[bias1])

        # Graph 2: Sub(bias, x) — same op, swapped inputs
        x2 = ir.val("x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 4]))
        bias_t2 = ir.Tensor(np.ones(4, dtype=np.float32), name="b")
        bias2 = ir.Value(
            name="b",
            const_value=bias_t2,
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([4]),
        )
        sub2 = ir.Node("", "Sub", [bias2, x2], name="sub")
        out2 = sub2.outputs[0]
        out2.name = "y"
        g2 = ir.Graph([x2], [out2], nodes=[sub2], initializers=[bias2])

        changes = diff_graphs(canonicalize_graph(g1), canonicalize_graph(g2))
        types = {c["type"] for c in changes}
        assert "changed_connectivity" in types


# ------------------------------------------------------------------
# render_markdown tests
# ------------------------------------------------------------------


class TestRenderMarkdown:
    """Tests for render_markdown."""

    def test_no_changes(self) -> None:
        md = render_markdown({})
        assert "Architecture Diff" in md
        assert "No architecture changes" in md
        assert "<!-- arch-diff-bot -->" in md

    def test_with_changes(self) -> None:
        diffs = {
            "llama": {
                "model": {
                    "changes": [
                        {
                            "type": "added_node",
                            "details": "+ Relu",
                        },
                    ],
                    "_base_ops": ["Add"],
                    "_head_ops": ["Add", "Relu"],
                    "_base_node_count": 1,
                    "_head_node_count": 2,
                }
            }
        }
        md = render_markdown(diffs)
        assert "Architecture Diff" in md
        # Summary table present
        assert "| llama" in md
        assert "1 |" in md
        # Details section present
        assert "<details>" in md
        assert "Added nodes" in md
        assert "Relu" in md
        # Legend present
        assert "Legend" in md

    def test_no_change_model_shows_no_change_emoji(self) -> None:
        diffs = {
            "bert": {
                "model": {
                    "changes": [],
                    "_base_ops": ["Add"],
                    "_head_ops": ["Add"],
                    "_base_node_count": 1,
                    "_head_node_count": 1,
                }
            }
        }
        md = render_markdown(diffs)
        assert "⚪" in md

    def test_interface_change_is_major(self) -> None:
        diffs = {
            "qwen2": {
                "model": {
                    "changes": [
                        {
                            "type": "interface_change",
                            "details": "output count 1 → 2",
                        },
                    ],
                    "_base_ops": [],
                    "_head_ops": [],
                    "_base_node_count": 0,
                    "_head_node_count": 0,
                }
            }
        }
        md = render_markdown(diffs)
        assert "🔴" in md

    def test_empty_diffs_dict(self) -> None:
        """Empty diffs dict produces no-change output."""
        md = render_markdown({})
        assert "No architecture changes" in md
        # Summary table header still present
        assert "| Model |" in md
        # No details sections
        assert "<details>" not in md

    def test_commit_shas_displayed(self) -> None:
        """Passing base_ref and head_ref shows comparison line without links."""
        md = render_markdown({}, base_ref="abc1234", head_ref="def5678")
        assert "Comparing `abc1234` → `def5678`" in md

    def test_commit_shas_as_links_when_repo_url_provided(self) -> None:
        """Passing repo_url renders SHAs as clickable GitHub links."""
        md = render_markdown(
            {},
            base_ref="abc1234",
            head_ref="def5678",
            repo_url="https://github.com/onnxruntime/mobius",
        )
        assert "[`abc1234`](https://github.com/onnxruntime/mobius/commit/abc1234)" in md
        assert "[`def5678`](https://github.com/onnxruntime/mobius/commit/def5678)" in md

    def test_commit_shas_omitted_when_not_provided(self) -> None:
        """Without refs, no comparison line appears."""
        md = render_markdown({})
        assert "Comparing" not in md

    def test_multiple_models(self) -> None:
        """Multiple model types are each rendered in the summary table."""
        diffs = {
            "llama": {
                "model": {
                    "changes": [
                        {"type": "added_node", "details": "+ Relu"},
                    ],
                    "_base_ops": ["Add"],
                    "_head_ops": ["Add", "Relu"],
                    "_base_node_count": 1,
                    "_head_node_count": 2,
                }
            },
            "bert": {
                "model": {
                    "changes": [
                        {"type": "changed_attrs", "details": "node[0] Concat: axis: 0 → 1"},
                    ],
                    "_base_ops": ["Concat"],
                    "_head_ops": ["Concat"],
                    "_base_node_count": 1,
                    "_head_node_count": 1,
                }
            },
        }
        md = render_markdown(diffs)
        assert "| llama" in md
        assert "| bert" in md
        # Both detail sections rendered (sorted alphabetically: bert < llama)
        assert "bert / model" in md
        assert "llama / model" in md
        # Different severity emojis
        assert "🟡" in md  # llama: moderate (added_node)
        assert "🔵" in md  # bert: minor (changed_attrs)

    def test_multiple_sub_models(self) -> None:
        """Multiple sub-models within one model type."""
        diffs = {
            "whisper": {
                "encoder": {
                    "changes": [],
                    "_base_ops": ["Add"],
                    "_head_ops": ["Add"],
                    "_base_node_count": 1,
                    "_head_node_count": 1,
                },
                "decoder": {
                    "changes": [
                        {"type": "added_node", "details": "+ Softmax"},
                    ],
                    "_base_ops": ["Add"],
                    "_head_ops": ["Add", "Softmax"],
                    "_base_node_count": 1,
                    "_head_node_count": 2,
                },
            }
        }
        md = render_markdown(diffs)
        assert "| whisper | decoder" in md
        assert "| whisper | encoder" in md
        # encoder has no changes → no-change emoji
        assert "⚪" in md


# ------------------------------------------------------------------
# Integration: canonicalize → diff round-trip
# ------------------------------------------------------------------


class TestRoundTrip:
    """End-to-end tests combining canonicalize + diff."""

    def test_identical_graph_round_trip(self) -> None:
        g = _add_relu_graph()
        canon = canonicalize_graph(g)
        assert diff_graphs(canon, canon) == []

    def test_structural_change_detected(self) -> None:
        base = _simple_add_graph()
        head = _add_relu_graph()
        changes = diff_graphs(canonicalize_graph(base), canonicalize_graph(head))
        assert len(changes) > 0


# ------------------------------------------------------------------
# Subgraph recursion — helpers (graphs containing an If with subgraphs)
# ------------------------------------------------------------------


def _if_graph(
    *,
    then_op: str = "Add",
    else_op: str = "Sub",
    then_axis: int | None = None,
    then_domain: str = "",
    then_node_name: str = "then_op",
    else_node_name: str = "else_op",
) -> ir.Graph:
    """Return a graph: y = If(cond) {then: then_op(x,x)} {else: else_op(x,x)}.

    ``then_axis`` (when given) attaches an ``axis`` attribute to the
    then-branch node so inner-attribute deltas can be exercised.
    ``then_domain`` sets the then-branch node's op domain (diff_graphs does
    not compare domain, so this exercises the 'subgraph changed' fallback).
    ``*_node_name`` allow renaming inner nodes to prove names are ignored.
    """
    x = ir.val("x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 4]))
    cond = ir.val("cond", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([]))

    then_attrs = [ir.AttrInt64("axis", then_axis)] if then_axis is not None else []
    then_node = ir.Node(then_domain, then_op, [x, x], then_attrs, name=then_node_name)
    then_out = then_node.outputs[0]
    then_out.name = "then_out"
    then_g = ir.Graph([], [then_out], nodes=[then_node], name="then_branch")

    else_node = ir.Node("", else_op, [x, x], name=else_node_name)
    else_out = else_node.outputs[0]
    else_out.name = "else_out"
    else_g = ir.Graph([], [else_out], nodes=[else_node], name="else_branch")

    if_node = ir.Node(
        "",
        "If",
        [cond],
        [ir.AttrGraph("then_branch", then_g), ir.AttrGraph("else_branch", else_g)],
        name="if_node",
    )
    if_out = if_node.outputs[0]
    if_out.name = "y"
    return ir.Graph([x, cond], [if_out], nodes=[if_node])


# ------------------------------------------------------------------
# Subgraph recursion — tests (architect D12: descend into subgraphs)
# ------------------------------------------------------------------


def _nested_if_graph(
    *, inner_then_op: str = "Add", inner_then_axis: int | None = None
) -> ir.Graph:
    """Return a graph whose If then-branch itself contains an If (depth 2).

    ``inner_then_axis`` attaches an ``axis`` attribute to the innermost
    then-branch node, to exercise a *pure inner-attribute* delta nested
    two levels deep (which must stay MINOR, not promote to structural).
    """
    x = ir.val("x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 4]))
    cond = ir.val("cond", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([]))

    # Innermost two branches.
    inner_then_attrs = (
        [ir.AttrInt64("axis", inner_then_axis)] if inner_then_axis is not None else []
    )
    inner_then = ir.Node("", inner_then_op, [x, x], inner_then_attrs, name="inner_then")
    it_out = inner_then.outputs[0]
    it_out.name = "it_out"
    inner_then_g = ir.Graph([], [it_out], nodes=[inner_then], name="inner_then_branch")

    inner_else = ir.Node("", "Sub", [x, x], name="inner_else")
    ie_out = inner_else.outputs[0]
    ie_out.name = "ie_out"
    inner_else_g = ir.Graph([], [ie_out], nodes=[inner_else], name="inner_else_branch")

    inner_if = ir.Node(
        "",
        "If",
        [cond],
        [
            ir.AttrGraph("then_branch", inner_then_g),
            ir.AttrGraph("else_branch", inner_else_g),
        ],
        name="inner_if",
    )
    inner_if_out = inner_if.outputs[0]
    inner_if_out.name = "inner_if_out"
    # Outer then-branch wraps the inner If; outer else-branch is a plain op.
    outer_then_g = ir.Graph([], [inner_if_out], nodes=[inner_if], name="outer_then")

    outer_else = ir.Node("", "Mul", [x, x], name="outer_else")
    oe_out = outer_else.outputs[0]
    oe_out.name = "oe_out"
    outer_else_g = ir.Graph([], [oe_out], nodes=[outer_else], name="outer_else")

    outer_if = ir.Node(
        "",
        "If",
        [cond],
        [
            ir.AttrGraph("then_branch", outer_then_g),
            ir.AttrGraph("else_branch", outer_else_g),
        ],
        name="outer_if",
    )
    out = outer_if.outputs[0]
    out.name = "y"
    return ir.Graph([x, cond], [out], nodes=[outer_if])


def _graphs_attr_graph(*, body_ops: list[str]) -> ir.Graph:
    """Return a graph with a node carrying a GRAPHS-plural attribute.

    GRAPHS is rare for standard ops (``If`` / ``Loop`` / ``Scan`` bodies are
    GRAPH singular), so this uses a synthetic op purely to exercise the
    plural recursion branch in ``_attr_to_comparable``.
    """
    x = ir.val("x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 4]))

    bodies = []
    for idx, op in enumerate(body_ops):
        n = ir.Node("", op, [x, x], name=f"body_{idx}")
        o = n.outputs[0]
        o.name = f"body_out_{idx}"
        bodies.append(ir.Graph([], [o], nodes=[n], name=f"body_g_{idx}"))

    multi = ir.Node("", "CustomMulti", [x], [ir.AttrGraphs("bodies", bodies)], name="cm")
    out = multi.outputs[0]
    out.name = "y"
    return ir.Graph([x], [out], nodes=[multi])


def _if_graph_branch_output_dtype(*, then_dtype: ir.DataType) -> ir.Graph:
    """If-graph whose then-branch *output dtype* varies (subgraph interface)."""
    x = ir.val("x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 4]))
    cond = ir.val("cond", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([]))

    cast = ir.Node("", "Cast", [x], [ir.AttrInt64("to", int(then_dtype))], name="cast")
    then_out = cast.outputs[0]
    then_out.name = "then_out"
    then_out.type = ir.TensorType(then_dtype)
    then_out.shape = ir.Shape([1, 4])
    then_g = ir.Graph([], [then_out], nodes=[cast], name="then_branch")

    else_node = ir.Node("", "Identity", [x], name="else_node")
    else_out = else_node.outputs[0]
    else_out.name = "else_out"
    else_out.type = ir.TensorType(ir.DataType.FLOAT)
    else_out.shape = ir.Shape([1, 4])
    else_g = ir.Graph([], [else_out], nodes=[else_node], name="else_branch")

    if_node = ir.Node(
        "",
        "If",
        [cond],
        [ir.AttrGraph("then_branch", then_g), ir.AttrGraph("else_branch", else_g)],
        name="if_node",
    )
    if_out = if_node.outputs[0]
    if_out.name = "y"
    return ir.Graph([x, cond], [if_out], nodes=[if_node])


class TestSubgraphRecursion:
    """canonicalize_graph / diff_graphs descend into GRAPH-typed attrs."""

    def test_canonicalize_descends_into_subgraphs(self) -> None:
        """GRAPH attrs are recursively canonicalised, not collapsed to a type."""
        canon = canonicalize_graph(_if_graph(then_op="Add", else_op="Sub"))
        if_attrs = canon["nodes"][0]["attributes"]
        # Not collapsed to the old "<GRAPH>" placeholder.
        assert if_attrs["then_branch"] != "<GRAPH>"
        assert if_attrs["else_branch"] != "<GRAPH>"
        # Nested canonical form carries the subgraph's op sequence.
        then_sub = if_attrs["then_branch"][_SUBGRAPH_KEY]
        else_sub = if_attrs["else_branch"][_SUBGRAPH_KEY]
        assert then_sub["op_sequence"] == ["Add"]
        assert else_sub["op_sequence"] == ["Sub"]

    def test_identical_subgraphs_no_diff(self) -> None:
        """Two structurally identical If-graphs produce no diff."""
        base = canonicalize_graph(_if_graph(then_op="Add", else_op="Sub"))
        head = canonicalize_graph(_if_graph(then_op="Add", else_op="Sub"))
        assert diff_graphs(base, head) == []

    def test_subgraph_node_names_ignored(self) -> None:
        """Renaming inner subgraph nodes does not produce a spurious diff."""
        base = canonicalize_graph(_if_graph(then_node_name="a", else_node_name="b"))
        head = canonicalize_graph(_if_graph(then_node_name="x", else_node_name="y"))
        assert diff_graphs(base, head) == []

    def test_then_branch_op_delta_is_structural(self) -> None:
        """A differing then-branch op is a MODERATE subgraph_structure_change."""
        base = canonicalize_graph(_if_graph(then_op="Add", else_op="Sub"))
        head = canonicalize_graph(_if_graph(then_op="Mul", else_op="Sub"))
        changes = diff_graphs(base, head)
        structural = [c for c in changes if c["type"] == "subgraph_structure_change"]
        assert structural, "op delta inside a branch must be structural"
        assert any("then_branch" in c["details"] for c in structural)
        # The op-level subgraph delta is surfaced (added/removed inner node).
        assert any("Add" in c["details"] and "Mul" in c["details"] for c in structural)

    def test_else_branch_op_delta_is_structural(self) -> None:
        """A differing else-branch op is detected independently of then-branch."""
        base = canonicalize_graph(_if_graph(then_op="Add", else_op="Sub"))
        head = canonicalize_graph(_if_graph(then_op="Add", else_op="Div"))
        changes = diff_graphs(base, head)
        structural = [c for c in changes if c["type"] == "subgraph_structure_change"]
        assert any("else_branch" in c["details"] for c in structural)

    def test_structural_subgraph_change_is_moderate(self) -> None:
        """render_markdown rates a subgraph structure change as MODERATE (🟡)."""
        base = canonicalize_graph(_if_graph(then_op="Add", else_op="Sub"))
        head = canonicalize_graph(_if_graph(then_op="Mul", else_op="Sub"))
        changes = diff_graphs(base, head)
        diffs = {
            "model": {
                "sub": {
                    "changes": changes,
                    "_base_ops": base["op_sequence"],
                    "_head_ops": head["op_sequence"],
                    "_base_node_count": len(base["nodes"]),
                    "_head_node_count": len(head["nodes"]),
                }
            }
        }
        md = render_markdown(diffs)
        assert "🟡" in md
        assert "Subgraph structure changes" in md

    def test_inner_attribute_delta_is_minor_and_surfaced(self) -> None:
        """A pure inner-attribute tweak (Concat axis) stays changed_attrs/MINOR.

        The actual nested detail (``axis: 0 → 1``) must be surfaced, not an
        opaque ``"N sub-change(s)"`` summary.
        """
        base = canonicalize_graph(_if_graph(then_op="Concat", else_op="Sub", then_axis=0))
        head = canonicalize_graph(_if_graph(then_op="Concat", else_op="Sub", then_axis=1))
        changes = diff_graphs(base, head)
        # Not promoted to structural — it's an inner-attr-only change.
        assert not any(c["type"] == "subgraph_structure_change" for c in changes)
        attr_changes = [c for c in changes if c["type"] == "changed_attrs"]
        assert attr_changes
        detail = " ".join(c["details"] for c in attr_changes)
        assert "then_branch" in detail
        # The real inner delta is surfaced.
        assert "axis" in detail and "0" in detail and "1" in detail

    def test_nested_subgraph_delta_detected(self) -> None:
        """A delta in an If-inside-an-If (depth 2) propagates to the top diff."""
        base = canonicalize_graph(_nested_if_graph(inner_then_op="Add"))
        head = canonicalize_graph(_nested_if_graph(inner_then_op="Mul"))
        changes = diff_graphs(base, head)
        structural = [c for c in changes if c["type"] == "subgraph_structure_change"]
        assert structural, "innermost branch op delta must surface at the top"
        assert any("then_branch" in c["details"] for c in structural)

    def test_nested_pure_attr_delta_stays_minor(self) -> None:
        """A pure inner-attr tweak nested two levels deep stays changed_attrs/MINOR.

        (Code-review NIT: lock the severity boundary — nesting must not
        spuriously promote a non-structural change to structural.)
        """
        base = canonicalize_graph(_nested_if_graph(inner_then_op="Concat", inner_then_axis=0))
        head = canonicalize_graph(_nested_if_graph(inner_then_op="Concat", inner_then_axis=1))
        changes = diff_graphs(base, head)
        assert not any(c["type"] == "subgraph_structure_change" for c in changes)
        attr_changes = [c for c in changes if c["type"] == "changed_attrs"]
        assert attr_changes
        # The innermost axis delta is still surfaced through both nesting levels.
        detail = " ".join(c["details"] for c in attr_changes)
        assert "then_branch" in detail and "axis" in detail

    def test_subgraph_changed_fallback_path(self) -> None:
        """A subgraph delta invisible to diff_graphs hits the readable fallback.

        (Code-review NIT: lock the ``"subgraph changed"`` fallback.)  The
        then-branch node's *domain* differs; canonicalize records domain so
        the attribute payloads differ, but diff_graphs does not compare
        domain, so the nested diff is empty → the summary falls back to
        ``"subgraph changed"`` and is classified MINOR (changed_attrs).
        """
        base = canonicalize_graph(_if_graph(then_op="Add", then_domain=""))
        head = canonicalize_graph(_if_graph(then_op="Add", then_domain="custom.domain"))
        changes = diff_graphs(base, head)
        assert not any(c["type"] == "subgraph_structure_change" for c in changes)
        attr_changes = [c for c in changes if c["type"] == "changed_attrs"]
        assert any("subgraph changed" in c["details"] for c in attr_changes)

    def test_graphs_plural_attr_recursed_and_diffed(self) -> None:
        """GRAPHS-plural attrs are canonicalised and per-body deltas detected."""
        base = canonicalize_graph(_graphs_attr_graph(body_ops=["Add", "Add"]))
        head = canonicalize_graph(_graphs_attr_graph(body_ops=["Add", "Mul"]))
        # Plural payload is recursed, not collapsed.
        bodies_attr = base["nodes"][0]["attributes"]["bodies"]
        assert bodies_attr != "<GRAPHS>"
        changes = diff_graphs(base, head)
        structural = [c for c in changes if c["type"] == "subgraph_structure_change"]
        assert structural
        # Plural payload genuinely drove the AttributeType.GRAPHS branch.
        assert _SUBGRAPHS_KEY in bodies_attr
        assert len(bodies_attr[_SUBGRAPHS_KEY]) == 2
        # The differing body is identified by index.
        assert any("subgraph[1]" in c["details"] for c in structural)

    def test_graphs_plural_branch_count_change_is_structural(self) -> None:
        """Adding/removing a body in a GRAPHS-plural attr is structural."""
        base = canonicalize_graph(_graphs_attr_graph(body_ops=["Add"]))
        head = canonicalize_graph(_graphs_attr_graph(body_ops=["Add", "Mul"]))
        changes = diff_graphs(base, head)
        structural = [c for c in changes if c["type"] == "subgraph_structure_change"]
        assert structural
        assert any("added" in c["details"] or "removed" in c["details"] for c in structural)

    def test_subgraph_interface_change_is_structural(self) -> None:
        """A subgraph *interface* (output dtype) change is structural."""
        base = canonicalize_graph(_if_graph_branch_output_dtype(then_dtype=ir.DataType.FLOAT))
        head = canonicalize_graph(
            _if_graph_branch_output_dtype(then_dtype=ir.DataType.FLOAT16)
        )
        changes = diff_graphs(base, head)
        structural = [c for c in changes if c["type"] == "subgraph_structure_change"]
        assert structural, "subgraph interface change must be structural"
        assert any("then_branch" in c["details"] for c in structural)

    def test_subgraph_delta_does_not_perturb_top_level_ops(self) -> None:
        """The top-level op sequence is unchanged when only a subgraph differs."""
        base = canonicalize_graph(_if_graph(then_op="Add", else_op="Sub"))
        head = canonicalize_graph(_if_graph(then_op="Mul", else_op="Sub"))
        # Both still have exactly one top-level If node.
        assert base["op_sequence"] == ["If"] == head["op_sequence"]
        changes = diff_graphs(base, head)
        # No spurious added/removed at the top level.
        assert not any(c["type"] in {"added_node", "removed_node"} for c in changes)

    def test_top_level_op_swap_not_double_counted(self) -> None:
        """An op_type swap at the same position is structural-only, never MINOR.

        (Reviewer 6def2895 (a): pick one classification — structural wins.)
        """

        def _single(op: str) -> ir.Graph:
            x = ir.val("x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 4]))
            n = ir.Node("", op, [x, x], name="n")
            o = n.outputs[0]
            o.name = "y"
            return ir.Graph([x], [o], nodes=[n])

        changes = diff_graphs(
            canonicalize_graph(_single("Add")), canonicalize_graph(_single("Mul"))
        )
        types = {c["type"] for c in changes}
        assert types == {"added_node", "removed_node"}
        assert "changed_attrs" not in types

    def test_subgraph_op_swap_is_structural_only(self) -> None:
        """MEA↔Flash-style op swap *inside* a branch is structural, not MINOR.

        A same-position op_type swap within a subgraph must classify as
        subgraph_structure_change only — not additionally as changed_attrs.
        """
        base = canonicalize_graph(_if_graph(then_op="Add", else_op="Sub"))
        head = canonicalize_graph(_if_graph(then_op="Mul", else_op="Sub"))
        changes = diff_graphs(base, head)
        # Exactly one change for the swapped branch, classified structural.
        assert [c["type"] for c in changes] == ["subgraph_structure_change"]
        assert not any(c["type"] == "changed_attrs" for c in changes)
