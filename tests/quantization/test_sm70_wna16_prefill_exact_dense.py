# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Gate and dispatch tests for the SM70 WNA16 exact-dense prefill route.

These run on CPU: the native operator is monkeypatched, and the route
contract under test is pure Python gate logic plus cuBLAS dispatch.
"""

import pytest
import torch

import vllm.envs as envs
from vllm import _sm70_ops as sm70_ops_module
from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
from vllm.scalar_type import scalar_types


@pytest.fixture(autouse=True)
def _clear_workspaces():
    sm70_tm.clear_sm70_turbomind_workspaces()
    yield
    sm70_tm.clear_sm70_turbomind_workspaces()


def _make_prepared_layer(
    k,
    n,
    *,
    group_size=128,
    op_kind="uint4",
    gated_silu=False,
    scales_dtype=torch.int32,
    tp_size=4,
    prefix="model.layers.0.mlp.gate_up_proj",
):
    layer = torch.nn.Module()
    layer.tp_size = tp_size
    layer.prefix = prefix
    state = sm70_tm.SM70TurboMindLinearState(
        weight=torch.zeros((max(k * n // 8, 1),), dtype=torch.int32),
        scales=torch.zeros((max(k // group_size, 1), n), dtype=scales_dtype),
        group_size=group_size,
        k_ld=0,
        q_ld=0,
        output_size=n,
        op_kind=op_kind,
        gated_silu=gated_silu,
    )
    setattr(layer, sm70_tm.STATE_ATTR, state)
    return layer


def _enable_route(monkeypatch, *, op_present=True):
    monkeypatch.delenv("VLLM_SM70_WNA16_PREFILL_EXACT_DENSE", raising=False)
    monkeypatch.setattr(sm70_tm, "_has_awq_dequantize_out_op", lambda: op_present)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    envs.disable_envs_cache()


def test_wna16_prefill_exact_dense_is_default_on(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_WNA16_PREFILL_EXACT_DENSE", raising=False)
    envs.disable_envs_cache()
    try:
        assert envs.VLLM_SM70_WNA16_PREFILL_EXACT_DENSE
    finally:
        envs.disable_envs_cache()


def test_attach_success_binds_shared_workspace(monkeypatch):
    _enable_route(monkeypatch)
    layer = _make_prepared_layer(5120, 8704)

    assert sm70_tm.attach_wna16_prefill_exact_dense(layer)
    state = getattr(layer, sm70_tm.STATE_ATTR)
    workspace = state.prefill_dense_workspace
    assert workspace is not None
    assert workspace.dtype == torch.float16
    assert workspace.numel() == max(
        k * n for k, n in sm70_tm._SM70_WNA16_PREFILL_DENSE_SHAPES.values()
    )
    # A second layer on the same device reuses the bounded allocation.
    other = _make_prepared_layer(4352, 5120, prefix="model.layers.1.mlp.down_proj")
    assert sm70_tm.attach_wna16_prefill_exact_dense(other)
    assert getattr(other, sm70_tm.STATE_ATTR).prefill_dense_workspace is workspace


def test_attach_env_off_keeps_turbomind_path(monkeypatch):
    _enable_route(monkeypatch)
    monkeypatch.setenv("VLLM_SM70_WNA16_PREFILL_EXACT_DENSE", "0")
    envs.disable_envs_cache()
    try:
        layer = _make_prepared_layer(5120, 8704)
        assert not sm70_tm.attach_wna16_prefill_exact_dense(layer)
        assert getattr(layer, sm70_tm.STATE_ATTR).prefill_dense_workspace is None
    finally:
        envs.disable_envs_cache()


def test_attach_requires_dequantize_op(monkeypatch):
    _enable_route(monkeypatch, op_present=False)
    layer = _make_prepared_layer(5120, 8704)
    assert not sm70_tm.attach_wna16_prefill_exact_dense(layer)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"op_kind": "mxfp4"},
        {"gated_silu": True},
        {"group_size": 32},
        {"tp_size": 2},
        {"scales_dtype": torch.uint8},
    ],
)
def test_attach_rejects_ineligible_states(monkeypatch, kwargs):
    _enable_route(monkeypatch)
    layer = _make_prepared_layer(5120, 8704, **kwargs)
    assert not sm70_tm.attach_wna16_prefill_exact_dense(layer)
    assert getattr(layer, sm70_tm.STATE_ATTR).prefill_dense_workspace is None


def test_attach_rejects_unknown_prefix_and_shape_mismatch(monkeypatch):
    _enable_route(monkeypatch)
    unknown = _make_prepared_layer(5120, 8704, prefix="model.layers.0.mlp.other")
    assert not sm70_tm.attach_wna16_prefill_exact_dense(unknown)
    mismatch = _make_prepared_layer(5120, 8704, prefix="model.layers.0.mlp.down_proj")
    assert not sm70_tm.attach_wna16_prefill_exact_dense(mismatch)


def test_clear_workspaces_releases_both_caches():
    sm70_tm._nvfp4_qpn4_dense_workspaces[(0, torch.float16)] = torch.empty(1)
    sm70_tm._wna16_prefill_dense_workspaces[(0, torch.float16)] = torch.empty(1)
    sm70_tm.clear_sm70_turbomind_workspaces()
    assert not sm70_tm._nvfp4_qpn4_dense_workspaces
    assert not sm70_tm._wna16_prefill_dense_workspaces


def test_wna16_scheme_wires_attach_after_prepare(monkeypatch):
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (  # noqa: E501
        compressed_tensors_wNa16 as wna16_module,
    )

    scheme = object.__new__(wna16_module.CompressedTensorsWNA16)
    scheme.symmetric = False
    scheme.group_size = 128
    scheme.has_g_idx = False
    scheme.quant_type = scalar_types.uint4
    layer = _make_prepared_layer(5120, 8704)
    layer.register_parameter(
        "weight_packed",
        torch.nn.Parameter(torch.empty(0, dtype=torch.int32), requires_grad=False),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(torch.empty(0, dtype=torch.float16), requires_grad=False),
    )
    layer.register_parameter(
        "weight_zero_point",
        torch.nn.Parameter(torch.empty(0, dtype=torch.int32), requires_grad=False),
    )

    calls = {}
    monkeypatch.setattr(
        wna16_module.sm70_tm,
        "should_prepare_turbomind",
        lambda tensor, enabled: True,
    )
    monkeypatch.setattr(
        wna16_module.sm70_tm,
        "prepare_compressed_uint4_linear",
        lambda layer, group_size, symmetric: calls.setdefault("prepare", True),
    )
    monkeypatch.setattr(
        wna16_module.sm70_tm,
        "attach_wna16_prefill_exact_dense",
        lambda prepared: calls.setdefault("attach", prepared) is prepared,
    )

    scheme.process_weights_after_loading(layer)

    assert calls["prepare"]
    assert calls["attach"] is layer


def _make_dispatch_layer(k, n, workspace=None):
    layer = torch.nn.Module()
    state = sm70_tm.SM70TurboMindLinearState(
        weight=torch.zeros((max(k * n // 8, 1),), dtype=torch.int32),
        scales=torch.zeros((max(k // 128, 1), n), dtype=torch.int32),
        group_size=128,
        k_ld=0,
        q_ld=0,
        output_size=n,
        op_kind="uint4",
    )
    if workspace is not None:
        state.prefill_dense_workspace = workspace
    setattr(layer, sm70_tm.STATE_ATTR, state)
    return layer


def _seed_dispatch_workspace(monkeypatch, k, n):
    """Seed the module cache the registered op resolves the workspace from."""
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    sm70_tm._wna16_prefill_dense_workspaces[(0, torch.float16)] = torch.empty(
        k * n, dtype=torch.float16
    )


def _install_recorders(monkeypatch):
    calls = {}

    def fake_dequant(out, packed_weight, packed_scales, group_size):
        calls["dequant"] = group_size
        out.fill_(0.5)

    def fake_tm_gemm(out, *args):
        calls["turbomind"] = args
        out.fill_(2.0)

    monkeypatch.setattr(sm70_ops_module, "awq_sm70_dequantize_out", fake_dequant)
    monkeypatch.setattr(sm70_ops_module, "awq_gemm_sm70_out", fake_tm_gemm)
    return calls


def test_dispatch_dense_route_above_threshold_with_bias(monkeypatch):
    k, n = 128, 64
    layer = _make_dispatch_layer(k, n, torch.empty(k * n, dtype=torch.float16))
    _seed_dispatch_workspace(monkeypatch, k, n)
    calls = _install_recorders(monkeypatch)

    x = torch.randn(2048, k, dtype=torch.float16)
    bias = torch.randn(n, dtype=torch.float16)
    out = sm70_tm.apply_prepared_linear(layer, x, bias)

    assert out.shape == (2048, n)
    assert calls["dequant"] == 128
    assert "turbomind" not in calls
    expected = x @ torch.full((k, n), 0.5, dtype=torch.float16) + bias
    assert torch.allclose(out, expected)


def test_dispatch_dense_route_preserves_leading_dims(monkeypatch):
    k, n = 128, 64
    layer = _make_dispatch_layer(k, n, torch.empty(k * n, dtype=torch.float16))
    _seed_dispatch_workspace(monkeypatch, k, n)
    calls = _install_recorders(monkeypatch)

    x = torch.randn(1, 4096, k, dtype=torch.float16)
    out = sm70_tm.apply_prepared_linear(layer, x, None)

    assert out.shape == (1, 4096, n)
    assert "dequant" in calls
    assert "turbomind" not in calls


def test_dispatch_dense_route_pads_kernel_output(monkeypatch):
    k, n = 128, 64
    layer = _make_dispatch_layer(k, n, torch.empty(k * n, dtype=torch.float16))
    _seed_dispatch_workspace(monkeypatch, k, n)
    calls = _install_recorders(monkeypatch)
    getattr(layer, sm70_tm.STATE_ATTR).padded_output_size = n + 32

    x = torch.randn(2048, k, dtype=torch.float16)
    out = sm70_tm.apply_prepared_linear(layer, x, None)

    assert out.shape == (2048, n)
    assert calls["dequant"] == 128
    assert "turbomind" not in calls
    expected = x @ torch.full((k, n), 0.5, dtype=torch.float16)
    assert torch.allclose(out, expected)


def test_dense_prefill_mm_op_fake_shape_matches_real():
    from torch._subclasses import FakeTensorMode

    with FakeTensorMode():
        x = torch.empty(7, 128, dtype=torch.float16)
        weight = torch.empty(16, dtype=torch.int32)
        scales = torch.empty(1, 64, dtype=torch.int32)
        out = torch.ops.sm70_tm.wna16_dense_prefill_mm(
            x, weight, scales, 128, 0, 0, 64, 96, 2048
        )

    assert out.shape == (7, 96)
    assert out.dtype == torch.float16


@pytest.mark.parametrize(
    "m,x_dtype",
    [
        (2047, torch.float16),
        (2048, torch.float32),
        (32, torch.float16),
    ],
)
def test_dispatch_keeps_turbomind_outside_route(monkeypatch, m, x_dtype):
    k, n = 128, 64
    layer = _make_dispatch_layer(k, n, torch.empty(k * n, dtype=torch.float16))
    _seed_dispatch_workspace(monkeypatch, k, n)
    calls = _install_recorders(monkeypatch)

    x = torch.randn(m, k, dtype=x_dtype)
    out = sm70_tm.apply_prepared_linear(layer, x, None)

    assert out.shape == (m, n)
    assert "turbomind" in calls
    assert "dequant" not in calls
    assert torch.allclose(out, torch.full_like(out, 2.0))


def test_dispatch_without_workspace_stays_on_turbomind(monkeypatch):
    k, n = 128, 64
    layer = _make_dispatch_layer(k, n)
    calls = _install_recorders(monkeypatch)

    x = torch.randn(4096, k, dtype=torch.float16)
    out = sm70_tm.apply_prepared_linear(layer, x, None)

    assert out.shape == (4096, n)
    assert "turbomind" in calls
    assert "dequant" not in calls
