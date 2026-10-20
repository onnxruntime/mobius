# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the MambaBlock component."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
from onnxscript import GraphBuilder

from mobius._constants import OPSET_VERSION
from mobius._testing import (
    count_op_type,
    create_test_builder,
    create_test_input,
)
from mobius.components import GroupedGatedRMSNorm, Mamba2Block
from mobius.components._mamba_block import MambaBlock, SequenceMambaBlock


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_mamba2_recurrence_precision_boundaries(dtype):
    """Protect export precision; native BF16 fallback has different cast boundaries."""
    from mobius._builder import _cast_module_dtype

    block = Mamba2Block(4, 4, 2, 2, 3, time_step_min=0.001)
    _cast_module_dtype(block, dtype)
    builder, op, graph = create_test_builder()
    hidden = builder.input("hidden", dtype=dtype, shape=[2, "seq", 4])
    conv = builder.input("conv", dtype=dtype, shape=[2, 10, 3])
    state = builder.input("state", dtype=ir.DataType.FLOAT, shape=[2, 2, 3, 2])
    block(op, hidden, conv, state)

    softplus = next(node for node in graph if node.op_type == "Softplus")
    dt_add = softplus.inputs[0].producer()
    assert dt_add.op_type == "Add"
    # In contrast to the native BF16 fallback, both operands are widened BEFORE
    # bias addition and softplus; this is not a claim of native backend parity.
    for value in dt_add.inputs:
        cast = value.producer()
        assert cast.op_type == "Cast"
        assert cast.attributes["to"].as_int() == int(ir.DataType.FLOAT)
    clip = next(node for node in graph if node.op_type == "Clip")
    assert clip.inputs[0] is softplus.outputs[0]
    attention = next(node for node in graph if node.op_type == "LinearAttention")
    assert attention.inputs[3].producer().op_type == "Cast"
    assert attention.inputs[3].producer().attributes["to"].as_int() == int(ir.DataType.FLOAT)
    readout_cast = next(
        node
        for node in graph
        if node.op_type == "CastLike" and node.inputs[0] is attention.outputs[0]
    )
    skip_add = next(
        node
        for node in graph
        if node.op_type == "Add" and node.inputs[0] is readout_cast.outputs[0]
    )
    assert readout_cast.inputs[1] is hidden
    skip_cast = skip_add.inputs[1].producer()
    assert skip_cast.op_type == "CastLike" and skip_cast.inputs[1] is hidden
    assert skip_cast.inputs[0].producer().op_type == "Reshape"
    d_product = skip_cast.inputs[0].producer().inputs[0].producer()
    assert d_product.op_type == "Mul"
    assert d_product.inputs[0].producer().op_type == "Reshape"
    assert d_product.inputs[1].producer().inputs[0].producer().op_type == "Cast"


@pytest.mark.parametrize("n_groups", [1, 2])
@pytest.mark.parametrize("time_step_min", [0.0, 0.001])
def test_mamba2_prefill_and_cached_timestep_contract(tmp_path, n_groups, time_step_min):
    """Exercise the bounded portable recurrence, not native cached-backend parity.

    Transformers Nemotron-H a96730c8c97b8efbf35bbaf7f5da33ec99231a49
    clamps dt in mamba2_chunk_scan, but not mamba2_selective_state_update.
    The cached contrast below preserves that distinction without changing the
    exporter or silently substituting a different reference.
    """
    import onnxruntime as ort
    import torch
    import torch.nn.functional as torch_f
    from onnx_ir.passes.common import InlinePass

    from mobius.functions import causal_conv_nd_with_state, linear_attention
    from mobius.integrations._weight_loading import apply_weights

    d_model, heads, d_head, d_state, kernel = 4, 2, 2, 3, 3
    d_inner = heads * d_head
    conv_dim = d_inner + 2 * n_groups * d_state
    block = Mamba2Block(
        d_model,
        d_inner,
        heads,
        d_head,
        d_state,
        n_groups=n_groups,
        conv_kernel=kernel,
        time_step_min=time_step_min,
    )
    block.norm = GroupedGatedRMSNorm(d_inner, group_size=d_head, eps=1e-5)
    graph = ir.Graph(
        [], [], nodes=[], name="mamba2", opset_imports={"": OPSET_VERSION, "com.microsoft": 1}
    )
    builder = GraphBuilder(graph)
    hidden = builder.input("hidden", dtype=ir.DataType.FLOAT, shape=[2, "seq", d_model])
    conv = builder.input("conv", dtype=ir.DataType.FLOAT, shape=[2, conv_dim, kernel - 1])
    state = builder.input("state", dtype=ir.DataType.FLOAT, shape=[2, heads, d_state, d_head])
    out, new_conv, new_state = block(builder.op, hidden, conv, state)
    for value, name in [(out, "out"), (new_conv, "new_conv"), (new_state, "new_state")]:
        builder.add_output(value, name)
    timestep = next(node for node in graph if node.op_type == "Softplus").outputs[0]
    if time_step_min:
        timestep = next(node for node in graph if node.op_type == "Clip").outputs[0]
    builder.add_output(timestep, "dt")
    model = ir.Model(graph, ir_version=11)
    for function in [
        causal_conv_nd_with_state(
            kernel_size=kernel, channels=conv_dim, ndim=1, activation="silu"
        ),
        linear_attention(q_num_heads=heads, kv_num_heads=heads, update_rule="gated"),
    ]:
        model.functions[function.identifier()] = function

    rng = np.random.default_rng(43)
    weights = {
        name: torch.from_numpy(rng.normal(0, 0.3, list(p.shape)).astype(np.float32))
        for name, p in graph.initializers.items()
        if p.const_value is None
    }
    # First head is below the Lightning minimum; second is safely above it.
    weights["dt_bias"] = torch.tensor([-11.0, -3.0])
    weights["A_log"] = torch.tensor([3.0, 0.0])
    weights["D"] = torch.tensor([0.4, -0.3])
    weights["norm.weight"] = torch.tensor([0.7, 1.2, 0.9, 1.1])
    apply_weights(model, weights)
    InlinePass()(model)
    path = tmp_path / "mamba2.onnx"
    ir.save(model, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def reference(u, conv_state, ssm_state, minimum):
        gate, xbc, raw_dt = torch.split(
            u @ weights["in_proj.weight"].T, [d_inner, conv_dim, heads], dim=-1
        )
        history = torch.cat([conv_state, xbc.transpose(1, 2)], dim=-1)
        activated = torch_f.silu(
            torch_f.conv1d(
                history, weights["conv1d.weight"], weights["conv1d.bias"], groups=conv_dim
            )
        ).transpose(1, 2)
        x, b, c = torch.split(
            activated, [d_inner, n_groups * d_state, n_groups * d_state], dim=-1
        )
        dt = torch_f.softplus(raw_dt + weights["dt_bias"]).clamp(min=minimum)
        b = b.reshape(2, -1, n_groups, d_state).repeat_interleave(heads // n_groups, 2)
        c = c.reshape(2, -1, n_groups, d_state).repeat_interleave(heads // n_groups, 2)
        x = x.reshape(2, -1, heads, d_head)
        values = []
        for t in range(u.shape[1]):
            delta = dt[:, t, :, None, None]
            decay = torch.exp(-weights["A_log"].exp()[None, :, None, None] * delta)
            ssm_state = ssm_state * decay + delta * b[:, t, :, :, None] * x[:, t, :, None, :]
            readout = (ssm_state * c[:, t, :, :, None]).sum(-2)
            values.append(readout + weights["D"][None, :, None] * x[:, t])
        y = torch.stack(values, 1).reshape(2, -1, d_inner) * torch_f.silu(gate)
        grouped = y.reshape(2, -1, heads, d_head)
        y = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + 1e-5)
        y = y.reshape(2, -1, d_inner) * weights["norm.weight"]
        return y @ weights["out_proj.weight"].T, history[:, :, -kernel + 1 :], ssm_state, dt

    actual_conv = np.zeros((2, conv_dim, kernel - 1), np.float32)
    actual_state = np.zeros((2, heads, d_state, d_head), np.float32)
    ref_conv, ref_state = torch.from_numpy(actual_conv.copy()), torch.from_numpy(actual_state)
    contrast_conv, contrast_state = ref_conv.clone(), ref_state.clone()
    for step, length in enumerate([3, 1, 1, 1]):
        u = torch.from_numpy(rng.normal(size=(2, length, d_model)).astype(np.float32))
        expected, ref_conv, ref_state, dt = reference(u, ref_conv, ref_state, time_step_min)
        contrast, contrast_conv, contrast_state, _ = reference(
            u, contrast_conv, contrast_state, time_step_min if step == 0 else 0.0
        )
        actual, actual_conv, actual_state, actual_dt = session.run(
            None, {"hidden": u.numpy(), "conv": actual_conv, "state": actual_state}
        )
        for got, want in [
            (actual, expected),
            (actual_conv, ref_conv),
            (actual_state, ref_state),
            (actual_dt, dt),
        ]:
            np.testing.assert_allclose(got, want.numpy(), atol=2e-6, rtol=2e-5)
        if time_step_min:
            assert torch.all(dt[:, :, 0] == time_step_min)
            assert torch.all(dt[:, :, 1] > time_step_min)
        if step and time_step_min:
            assert not torch.allclose(ref_state, contrast_state, atol=1e-7, rtol=1e-5)
            assert not torch.allclose(expected, contrast, atol=1e-7, rtol=1e-5)
        else:
            torch.testing.assert_close(expected, contrast, atol=0, rtol=0)


class TestMambaBlock:
    """Tests for MambaBlock graph construction."""

    def test_default_dt_rank(self):
        """dt_rank defaults to ceil(d_model / 16)."""
        block = MambaBlock(d_model=64, d_inner=128)
        # ceil(64 / 16) = 4
        assert block.dt_rank == 4

    def test_default_dt_rank_non_divisible(self):
        """dt_rank rounds up for non-divisible d_model."""
        block = MambaBlock(d_model=100, d_inner=200)
        # ceil(100 / 16) = 7
        assert block.dt_rank == 7

    def test_custom_dt_rank(self):
        """dt_rank can be overridden."""
        block = MambaBlock(d_model=64, d_inner=128, dt_rank=8)
        assert block.dt_rank == 8

    def test_parameters_created(self):
        """All sub-module parameters are created."""
        block = MambaBlock(d_model=64, d_inner=128, d_state=16)
        params = list(block.parameters())
        # in_proj.weight, conv1d.weight, conv1d.bias,
        # ssm.x_proj.weight, ssm.dt_proj.weight, ssm.dt_proj.bias,
        # ssm.A_log, ssm.D, out_proj.weight
        assert len(params) == 9

    def test_in_proj_shape(self):
        """Input projection maps d_model → 2*d_inner."""
        block = MambaBlock(d_model=64, d_inner=128)
        # in_proj: (2*d_inner, d_model) = (256, 64)
        assert list(block.in_proj.weight.shape) == [256, 64]

    def test_out_proj_shape(self):
        """Output projection maps d_inner → d_model."""
        block = MambaBlock(d_model=64, d_inner=128)
        # out_proj: (d_model, d_inner) = (64, 128)
        assert list(block.out_proj.weight.shape) == [64, 128]

    def test_conv1d_shape(self):
        """Conv1D weight has correct shape."""
        block = MambaBlock(d_model=64, d_inner=128, conv_kernel=4)
        # conv1d.weight: (d_inner, 1, conv_kernel) = (128, 1, 4)
        assert list(block.conv1d.weight.shape) == [128, 1, 4]

    def test_forward_builds_graph(self):
        """Forward pass constructs a valid ONNX graph."""
        block = MambaBlock(d_model=64, d_inner=128, d_state=16, conv_kernel=4)
        test_builder, op, _graph = create_test_builder()
        hidden = create_test_input(test_builder, "hidden_states", [2, 1, 64])
        conv_state = create_test_input(test_builder, "conv_state", [2, 128, 3])
        ssm_state = create_test_input(test_builder, "ssm_state", [2, 128, 16])

        output, new_conv, new_ssm = block(op, hidden, conv_state, ssm_state)

        assert output is not None
        assert new_conv is not None
        assert new_ssm is not None

    def test_conv_op_present(self):
        """Graph contains Conv op from depthwise conv1d."""
        block = MambaBlock(d_model=32, d_inner=64, d_state=8)
        test_builder, op, graph = create_test_builder()
        hidden = create_test_input(test_builder, "hidden_states", [1, 1, 32])
        conv_state = create_test_input(test_builder, "conv_state", [1, 64, 3])
        ssm_state = create_test_input(test_builder, "ssm_state", [1, 64, 8])

        block(op, hidden, conv_state, ssm_state)

        assert count_op_type(graph, "Conv") >= 1

    def test_split_ops_present(self):
        """Graph contains Split ops for in_proj and SSM projections."""
        block = MambaBlock(d_model=32, d_inner=64, d_state=8)
        test_builder, op, graph = create_test_builder()
        hidden = create_test_input(test_builder, "hidden_states", [1, 1, 32])
        conv_state = create_test_input(test_builder, "conv_state", [1, 64, 3])
        ssm_state = create_test_input(test_builder, "ssm_state", [1, 64, 8])

        block(op, hidden, conv_state, ssm_state)

        # Two splits: in_proj (x/z) and SSM x_proj (dt/B/C)
        assert count_op_type(graph, "Split") >= 2

    def test_silu_activation_uses_fused_swish(self):
        """Both SiLU activations use the fused Swish op."""
        block = MambaBlock(d_model=32, d_inner=64, d_state=8)
        test_builder, op, graph = create_test_builder()
        hidden = create_test_input(test_builder, "hidden_states", [1, 1, 32])
        conv_state = create_test_input(test_builder, "conv_state", [1, 64, 3])
        ssm_state = create_test_input(test_builder, "ssm_state", [1, 64, 8])

        block(op, hidden, conv_state, ssm_state)

        # SiLU appears in the convolution path and output gate.
        assert count_op_type(graph, "Swish") == 2
        assert count_op_type(graph, "Sigmoid") == 0

    def test_matmul_ops_for_projections(self):
        """Graph contains MatMul ops for linear projections."""
        block = MambaBlock(d_model=32, d_inner=64, d_state=8)
        test_builder, op, graph = create_test_builder()
        hidden = create_test_input(test_builder, "hidden_states", [1, 1, 32])
        conv_state = create_test_input(test_builder, "conv_state", [1, 64, 3])
        ssm_state = create_test_input(test_builder, "ssm_state", [1, 64, 8])

        block(op, hidden, conv_state, ssm_state)

        # in_proj, x_proj, dt_proj, out_proj = at least 4 MatMul ops
        assert count_op_type(graph, "MatMul") >= 4

    def test_custom_conv_kernel(self):
        """Custom conv_kernel size is reflected in parameters."""
        block = MambaBlock(d_model=32, d_inner=64, conv_kernel=8)
        assert list(block.conv1d.weight.shape) == [64, 1, 8]
        assert block.conv_kernel == 8


class TestSequenceMambaBlock:
    """Tests for the full-sequence (stateless) Mamba1 block."""

    def test_parameters_match_decode_block(self):
        """Parameter names and shapes are identical to MambaBlock's."""

        def spec(module):
            return {name: list(p.shape) for name, p in module.named_parameters()}

        assert spec(SequenceMambaBlock(d_model=32, d_inner=64, d_state=8)) == spec(
            MambaBlock(d_model=32, d_inner=64, d_state=8)
        )

    def test_forward_needs_no_state(self):
        """The whole sequence is consumed in one call, with no carried state."""
        block = SequenceMambaBlock(d_model=32, d_inner=64, d_state=8)
        test_builder, op, graph = create_test_builder()
        hidden = create_test_input(test_builder, "hidden_states", [2, 7, 32])

        out = block(op, hidden)

        assert out is not None
        assert count_op_type(graph, "Scan") == 1
        # in_proj, x_proj, dt_proj, out_proj
        assert count_op_type(graph, "MatMul") >= 4

    def test_conv_is_left_padded_for_causality(self):
        """Conv1D is left-padded by kernel_size - 1 so it stays causal."""
        block = SequenceMambaBlock(d_model=16, d_inner=32, d_state=4, conv_kernel=4)
        test_builder, op, graph = create_test_builder()
        hidden = create_test_input(test_builder, "hidden_states", [1, 5, 16])

        block(op, hidden)

        pad = next(node for node in graph if node.op_type == "Pad")
        pads = pad.inputs[1].const_value.numpy().tolist()
        # (batch, d_inner, seq): 3 begin values then 3 end values.
        assert pads == [0, 0, 3, 0, 0, 0]

    def test_matches_reference_recurrence(self):
        """ONNX output matches an explicit PyTorch selective-scan reference."""
        pytest.importorskip("onnxruntime")
        import onnxruntime as ort
        import torch
        import torch.nn.functional as torch_f

        d_model, d_state, conv_kernel, expand = 8, 4, 4, 2
        d_inner = d_model * expand
        dt_rank = -(-d_model // 16)
        batch, seq_len = 2, 6

        torch.manual_seed(0)
        weights = {
            "in_proj.weight": torch.randn(2 * d_inner, d_model) * 0.2,
            "conv1d.weight": torch.randn(d_inner, 1, conv_kernel) * 0.2,
            "conv1d.bias": torch.randn(d_inner) * 0.2,
            "ssm.x_proj.weight": torch.randn(dt_rank + 2 * d_state, d_inner) * 0.2,
            "ssm.dt_proj.weight": torch.randn(d_inner, dt_rank) * 0.2,
            "ssm.dt_proj.bias": torch.randn(d_inner) * 0.2,
            "ssm.A_log": torch.randn(d_inner, d_state) * 0.2,
            "ssm.D": torch.randn(d_inner) * 0.2,
            "out_proj.weight": torch.randn(d_model, d_inner) * 0.2,
        }

        def reference(u):
            """``mamba_ssm.Mamba`` forward, written out with plain torch ops."""
            x, z = (u @ weights["in_proj.weight"].t()).chunk(2, dim=-1)
            x = torch_f.conv1d(
                torch_f.pad(x.transpose(1, 2), (conv_kernel - 1, 0)),
                weights["conv1d.weight"],
                weights["conv1d.bias"],
                groups=d_inner,
            )
            x = torch_f.silu(x).transpose(1, 2)

            x_dbl = x @ weights["ssm.x_proj.weight"].t()
            dt_raw, b_mat, c_mat = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=-1)
            dt = torch_f.softplus(
                dt_raw @ weights["ssm.dt_proj.weight"].t() + weights["ssm.dt_proj.bias"]
            )

            a_neg = -torch.exp(weights["ssm.A_log"])
            state = torch.zeros(u.shape[0], d_inner, d_state)
            outputs = []
            for t in range(u.shape[1]):
                dt_col = dt[:, t].unsqueeze(-1)
                decay = torch.exp(dt_col * a_neg.unsqueeze(0))
                update = dt_col * x[:, t].unsqueeze(-1) * b_mat[:, t].unsqueeze(1)
                state = decay * state + update
                outputs.append((state * c_mat[:, t].unsqueeze(1)).sum(-1))
            y = torch.stack(outputs, dim=1) + weights["ssm.D"] * x
            return (y * torch_f.silu(z)) @ weights["out_proj.weight"].t()

        graph = ir.Graph([], [], nodes=[], name="g", opset_imports={"": OPSET_VERSION})
        builder = GraphBuilder(graph)
        hidden = builder.input(
            "hidden_states", dtype=ir.DataType.FLOAT, shape=["batch", "seq", d_model]
        )
        block = SequenceMambaBlock(d_model, d_inner, d_state, dt_rank, conv_kernel)
        for name, param in block.named_parameters():
            param.const_value = ir.tensor(weights[name].numpy())
        builder.add_output(block(builder.op, hidden), "y")

        session = ort.InferenceSession(
            ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        hidden_states = torch.randn(batch, seq_len, d_model)
        (got,) = session.run(None, {"hidden_states": hidden_states.numpy()})

        np.testing.assert_allclose(got, reference(hidden_states).detach().numpy(), atol=1e-5)
