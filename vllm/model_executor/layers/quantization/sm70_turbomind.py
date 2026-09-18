# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Literal

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

U4_GROUP_SIZES = (32, 64, 128)
GPTQ_GROUP_SIZES = (128,)
COMPRESSED_UINT4_GROUP_SIZES = (32, 128)
MXFP4_GROUP_SIZE = 32
NVFP4_GROUP_SIZE = 16
# SM70 packed NVFP4 GEMM needs complete 32-column tiles. N=8240 (Qwen
# GDN on TP2) is 16-aligned but corrupts the result without this padding.
NVFP4_OUTPUT_ALIGNMENT = 32
NVFP4_QPN4_DENSE_WORKSPACE_ELEMENTS = 5120 * 8704
STATE_ATTR = "_sm70_turbomind_linear"
SM70QuantBackend = Literal["auto", "marlin", "turbomind"]


@dataclass
class SM70TurboMindLinearState:
    weight: torch.Tensor
    scales: torch.Tensor
    group_size: int
    k_ld: int
    q_ld: int
    output_size: int
    op_kind: Literal["uint4", "mxfp4", "nvfp4", "nvfp4_qpn4"]
    gated_silu: bool = False
    dense_weight_ptr: int = 0
    global_scale: float = 0.0
    use_scale_code: bool = False
    padded_output_size: int = 0
    prefill_dense_workspace: torch.Tensor | None = None


# Compressed-tensors W4A16 exact-dense prefill: bounded FP16 expansion of the
# TurboMind uint4 state, consumed by the shared AWQ dequant operator plus
# cuBLAS. Shapes are the Qwen3.8-27B TP4 per-rank prepared projections; all
# satisfy the dequant contract (group_size 128, K % 128 == 0, N % 32 == 0).
_SM70_WNA16_PREFILL_DENSE_MIN_M = 2048
# The runtime GDN input projection is one fused in_proj_qkvz module
# (per-rank N = (10240 + 6144) / 4) even though the Qwen3.8 checkpoint
# stores in_proj_qkv and in_proj_z as separate tensors.
_SM70_WNA16_PREFILL_DENSE_SHAPES = {
    "gate_up_proj": (5120, 8704),
    "down_proj": (4352, 5120),
    "in_proj_qkvz": (5120, 4096),
    "out_proj": (1536, 5120),
    "qkv_proj": (5120, 3584),
    "o_proj": (1536, 5120),
}
_SM70_WNA16_PREFILL_DENSE_WORKSPACE_ELEMENTS = max(
    k * n for k, n in _SM70_WNA16_PREFILL_DENSE_SHAPES.values()
)
_wna16_prefill_dense_workspaces: dict[tuple[int, torch.dtype], torch.Tensor] = {}
# One info line per admitted projection class keeps load-time route evidence
# observable without one line per layer.
_wna16_prefill_dense_admitted_suffixes: set[str] = set()


def _has_awq_dequantize_out_op() -> bool:
    return hasattr(torch.ops._C, "awq_sm70_dequantize_out")


def _get_wna16_prefill_dense_workspace(weight: torch.Tensor) -> torch.Tensor | None:
    device_index = weight.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    cache_key = (device_index, torch.float16)
    workspace = _wna16_prefill_dense_workspaces.get(cache_key)
    if workspace is not None:
        return workspace
    try:
        workspace = torch.empty(
            (_SM70_WNA16_PREFILL_DENSE_WORKSPACE_ELEMENTS,),
            dtype=torch.float16,
            device=weight.device,
        )
    except torch.OutOfMemoryError:
        logger.warning_once(
            "Insufficient memory for the bounded SM70 WNA16 prefill workspace; "
            "falling back to the TurboMind uint4 path."
        )
        return None
    _wna16_prefill_dense_workspaces[cache_key] = workspace
    return workspace


def attach_wna16_prefill_exact_dense(layer: torch.nn.Module) -> bool:
    """Attach the bounded-workspace exact-dense prefill route to one layer.

    Evaluates the load-time contract once, after
    :func:`prepare_compressed_uint4_linear`. On any failed condition the
    layer silently keeps the TurboMind uint4 path unchanged.
    """
    if not envs.VLLM_SM70_WNA16_PREFILL_EXACT_DENSE:
        return False
    if not _has_awq_dequantize_out_op():
        return False
    state = getattr(layer, STATE_ATTR, None)
    if state is None or state.op_kind != "uint4":
        return False
    if state.group_size != 128 or state.gated_silu:
        return False
    if getattr(layer, "tp_size", 1) != 4:
        return False
    suffix = getattr(layer, "prefix", "").rsplit(".", 1)[-1]
    expected = _SM70_WNA16_PREFILL_DENSE_SHAPES.get(suffix)
    if expected is None:
        return False
    scales = state.scales
    # The shared dequant operator consumes the non-compact TurboMind
    # statistics: int32 [k / group_size, n] with fused scale and zero words.
    if scales.dtype != torch.int32 or scales.dim() != 2:
        return False
    n = state.output_size
    k = scales.size(0) * state.group_size
    if scales.size(1) != n or (k, n) != expected:
        return False
    workspace = _get_wna16_prefill_dense_workspace(scales)
    if workspace is None:
        return False
    state.prefill_dense_workspace = workspace
    if suffix not in _wna16_prefill_dense_admitted_suffixes:
        _wna16_prefill_dense_admitted_suffixes.add(suffix)
        logger.info(
            "SM70 WNA16 exact-dense prefill attached: %s (K=%d, N=%d).",
            suffix,
            k,
            n,
        )
    logger.info_once(
        "SM70 compressed-tensors WNA16 exact-dense prefill path enabled "
        "with a bounded 85 MiB workspace."
    )
    return True


# States retain only data_ptr(), so this cache owns the bounded allocation.
_nvfp4_qpn4_dense_workspaces: dict[tuple[int, torch.dtype], torch.Tensor] = {}


def clear_sm70_turbomind_workspaces() -> None:
    """Release process-global NVFP4 QPN4 and WNA16 prefill workspaces."""
    _nvfp4_qpn4_dense_workspaces.clear()
    _wna16_prefill_dense_workspaces.clear()


def quant_backend() -> SM70QuantBackend:
    return envs.get_sm70_quant_backend()


def use_turbomind(default_enabled: bool) -> bool:
    return envs.use_sm70_turbomind(default_enabled)


def forces_marlin() -> bool:
    return envs.force_sm70_marlin()


def is_exact_sm70_cuda(tensor: torch.Tensor, enabled: bool) -> bool:
    if not enabled or not tensor.is_cuda:
        return False
    return torch.cuda.get_device_capability(tensor.device) == (7, 0)


def is_exact_sm70_cuda_platform() -> bool:
    """Return true only for Volta SM70 CUDA workers.

    Quant-method selection runs before a layer owns a CUDA tensor, so it
    cannot use :func:`is_exact_sm70_cuda`. Keep this platform check separate
    from the tensor-based helpers used by linear weight preparation.
    """
    return current_platform.is_cuda() and current_platform.is_device_capability((7, 0))


def should_use_mxfp4_moe_turbomind() -> bool:
    """Select the native MXFP4 MoE path only on exact SM70."""
    return is_exact_sm70_cuda_platform() and use_turbomind(
        envs.VLLM_SM70_MXFP4_TURBOMIND
    )


def should_use_nvfp4_moe_turbomind() -> bool:
    """Select the native NVFP4 MoE path only on exact SM70."""
    return is_exact_sm70_cuda_platform() and use_turbomind(
        envs.VLLM_SM70_NVFP4_TURBOMIND
    )


def should_prepare_turbomind(
    tensor: torch.Tensor,
    default_enabled: bool,
) -> bool:
    return is_exact_sm70_cuda(tensor, use_turbomind(default_enabled))


def should_prepare_turbomind_or_marlin(
    tensor: torch.Tensor,
    default_enabled: bool,
) -> bool:
    return is_exact_sm70_cuda(tensor, use_turbomind(default_enabled) or forces_marlin())


def _get_u4_slices(x: torch.Tensor, dtype: torch.dtype) -> list[torch.Tensor]:
    if x.dtype == torch.int32:
        count = 8
    elif x.dtype == torch.uint8:
        count = 2
    else:
        raise TypeError(f"expected int32 or uint8 packed int4 tensor, got {x.dtype}")
    xs = []
    for _ in range(count):
        xs.append((x & 15).to(dtype))
        x = x >> 4
    return xs


def unpack_gptq_weight(qweight: torch.Tensor) -> torch.Tensor:
    xs = _get_u4_slices(qweight, torch.uint8)
    return torch.stack(xs, dim=1).reshape(-1, qweight.size(-1)).contiguous()


def unpack_gptq_zeros(qzeros: torch.Tensor) -> torch.Tensor:
    xs = _get_u4_slices(qzeros, torch.uint8)
    zeros = torch.stack(xs, dim=-1).reshape(qzeros.size(0), -1)
    return (zeros + 1).to(torch.float16).contiguous()


def unpack_compressed_weight(weight_packed: torch.Tensor) -> torch.Tensor:
    xs = _get_u4_slices(weight_packed, torch.uint8)
    weight = torch.stack(xs, dim=-1).reshape(*weight_packed.shape[:-1], -1)
    return weight.t().contiguous()


def unpack_compressed_zeros(weight_zero_point: torch.Tensor) -> torch.Tensor:
    xs = _get_u4_slices(weight_zero_point, torch.uint8)
    zeros = torch.stack(xs, dim=1).reshape(-1, weight_zero_point.size(-1))
    return zeros.t().to(torch.float16).contiguous()


def unpack_mxfp4_weight(weight_packed: torch.Tensor) -> torch.Tensor:
    if weight_packed.dim() > 2:
        weight_packed = torch.flatten(weight_packed, start_dim=-2)
    xs = _get_u4_slices(weight_packed, torch.uint8)
    weight = torch.flatten(
        torch.stack(xs, dim=-1),
        start_dim=-2,
    )
    return weight.t().contiguous()


def symmetric_int4_zeros_like(scales: torch.Tensor) -> torch.Tensor:
    return torch.full_like(scales, 8, dtype=torch.float16)


def _store_state(
    layer: torch.nn.Module,
    weight: torch.Tensor,
    scales: torch.Tensor,
    meta: torch.Tensor | None,
    group_size: int,
    output_size: int,
    op_kind: Literal["uint4", "mxfp4", "nvfp4", "nvfp4_qpn4"],
    gated_silu: bool = False,
    dense_weight_ptr: int = 0,
    global_scale: float = 0.0,
    use_scale_code: bool = False,
    padded_output_size: int = 0,
) -> None:
    state = SM70TurboMindLinearState(
        weight=weight,
        scales=scales,
        group_size=group_size,
        k_ld=0 if meta is None else int(meta[0]),
        q_ld=0 if meta is None else int(meta[1]),
        output_size=output_size,
        op_kind=op_kind,
        gated_silu=gated_silu,
        dense_weight_ptr=dense_weight_ptr,
        global_scale=global_scale,
        use_scale_code=use_scale_code,
        padded_output_size=padded_output_size,
    )
    setattr(layer, STATE_ATTR, state)


def has_prepared_linear(layer: torch.nn.Module) -> bool:
    return getattr(layer, STATE_ATTR, None) is not None


def prepare_gptq_linear(
    layer: torch.nn.Module,
    group_size: int,
    interleave_gated_silu: bool = False,
) -> None:
    if group_size not in GPTQ_GROUP_SIZES:
        raise RuntimeError(
            f"SM70 TurboMind GPTQ supports group_size 128, but got {group_size}."
        )
    if not hasattr(torch.ops._C, "uint4_sm70_prepare"):
        raise RuntimeError(
            "VLLM_SM70_GPTQ_TURBOMIND=1 requires a build with CUDA arch 7.0 "
            "and the SM70 TurboMind extension."
        )
    from vllm import _sm70_ops as sm70_ops

    qweight = unpack_gptq_weight(layer.qweight.data)
    scales = layer.scales.data.to(torch.float16).contiguous()
    zeros = unpack_gptq_zeros(layer.qzeros.data)
    tm_weight, tm_scales, meta = sm70_ops.uint4_sm70_prepare(
        qweight, scales, zeros, group_size, interleave_gated_silu
    )
    _store_state(
        layer,
        tm_weight,
        tm_scales,
        meta,
        group_size,
        qweight.size(1),
        "uint4",
    )


def prepare_compressed_uint4_linear(
    layer: torch.nn.Module,
    group_size: int,
    symmetric: bool,
    interleave_gated_silu: bool = False,
) -> None:
    if group_size not in COMPRESSED_UINT4_GROUP_SIZES:
        raise RuntimeError(
            "SM70 TurboMind compressed-tensors int4 supports "
            f"group_size 32/128, but got {group_size}."
        )
    if not hasattr(torch.ops._C, "uint4_sm70_prepare"):
        raise RuntimeError(
            "VLLM_SM70_COMPRESSED_TENSORS_TURBOMIND=1 requires a build with "
            "CUDA arch 7.0 and the SM70 TurboMind extension."
        )
    from vllm import _sm70_ops as sm70_ops

    qweight = unpack_compressed_weight(layer.weight_packed.data)
    scales = layer.weight_scale.data.t().to(torch.float16).contiguous()
    if symmetric:
        zeros = symmetric_int4_zeros_like(scales)
    else:
        zeros = unpack_compressed_zeros(layer.weight_zero_point.data)
    tm_weight, tm_scales, meta = sm70_ops.uint4_sm70_prepare(
        qweight, scales, zeros, group_size, interleave_gated_silu
    )
    _store_state(
        layer,
        tm_weight,
        tm_scales,
        meta,
        group_size,
        qweight.size(1),
        "uint4",
    )


def prepare_mxfp4_linear(
    layer: torch.nn.Module,
    interleave_gated_silu: bool = False,
) -> None:
    if not hasattr(torch.ops._C, "mxfp4_sm70_prepare"):
        raise RuntimeError(
            "VLLM_SM70_MXFP4_TURBOMIND=1 requires a build with CUDA arch 7.0 "
            "and the SM70 TurboMind extension."
        )
    from vllm import _sm70_ops as sm70_ops

    qweight = unpack_mxfp4_weight(layer.weight_packed.data)
    scales = layer.weight_scale.data.t().contiguous()
    tm_weight, tm_scales, meta = sm70_ops.mxfp4_sm70_prepare(
        qweight, scales, MXFP4_GROUP_SIZE, interleave_gated_silu
    )
    _store_state(
        layer,
        tm_weight,
        tm_scales,
        meta,
        MXFP4_GROUP_SIZE,
        qweight.size(1),
        "mxfp4",
    )


def prepare_nvfp4_linear(
    layer: torch.nn.Module,
    interleave_gated_silu: bool = False,
) -> None:
    if not hasattr(torch.ops._C, "nvfp4_sm70_prepare"):
        raise RuntimeError(
            "VLLM_SM70_NVFP4_TURBOMIND=1 requires a build with CUDA arch 7.0 "
            "and the SM70 TurboMind NVFP4 extension."
        )
    from vllm import _sm70_ops as sm70_ops

    qweight = unpack_mxfp4_weight(layer.weight.data)
    scales = (
        (
            layer.weight_scale.data.t().to(torch.float32)
            * layer.weight_global_scale.to(torch.float32)
        )
        .to(torch.float16)
        .contiguous()
    )
    output_size = qweight.size(1)
    padded_output_size = (
        (output_size + NVFP4_OUTPUT_ALIGNMENT - 1) // NVFP4_OUTPUT_ALIGNMENT
    ) * NVFP4_OUTPUT_ALIGNMENT
    if padded_output_size != output_size:
        if interleave_gated_silu:
            raise RuntimeError(
                "SM70 TurboMind NVFP4 gated-SiLU does not support output padding."
            )
        padded_qweight = torch.zeros(
            (qweight.size(0), padded_output_size),
            dtype=qweight.dtype,
            device=qweight.device,
        )
        padded_scales = torch.zeros(
            (scales.size(0), padded_output_size),
            dtype=scales.dtype,
            device=scales.device,
        )
        padded_qweight[:, :output_size].copy_(qweight)
        padded_scales[:, :output_size].copy_(scales)
        qweight = padded_qweight
        scales = padded_scales
    tm_weight, tm_scales, meta = sm70_ops.nvfp4_sm70_prepare(
        qweight, scales, NVFP4_GROUP_SIZE, interleave_gated_silu
    )
    _store_state(
        layer,
        tm_weight,
        tm_scales,
        meta,
        NVFP4_GROUP_SIZE,
        output_size,
        "nvfp4",
        interleave_gated_silu,
        padded_output_size=padded_output_size,
    )


def get_nvfp4_qpn4_dense_workspace(weight: torch.Tensor) -> torch.Tensor | None:
    device_index = weight.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    cache_key = (device_index, torch.float16)
    workspace = _nvfp4_qpn4_dense_workspaces.get(cache_key)
    if workspace is not None:
        return workspace
    try:
        workspace = torch.empty(
            (NVFP4_QPN4_DENSE_WORKSPACE_ELEMENTS,),
            dtype=torch.float16,
            device=weight.device,
        )
    except torch.OutOfMemoryError:
        return None
    _nvfp4_qpn4_dense_workspaces[cache_key] = workspace
    return workspace


def prepare_nvfp4_qpn4_linear(
    layer: torch.nn.Module,
    workspace: torch.Tensor,
    gated_silu: bool,
) -> None:
    """Replace one accepted NVFP4 dense weight with memory-neutral QPN4."""
    from vllm import _sm70_ops as sm70_ops

    qweight = unpack_mxfp4_weight(layer.weight.data)
    use_scale_code = gated_silu or envs.VLLM_SM70_NVFP4_QPN4_DOWN_SCALE_CODE
    global_scale = 0.0
    if use_scale_code:
        global_scale = float(layer.weight_global_scale.detach().float().item())
        raw_scale_codes = layer.weight_scale.data.t().contiguous()
        packed_weight, packed_scales = sm70_ops.nvfp4_qpn4_prepare_scale_code_sm70(
            qweight, raw_scale_codes
        )
    else:
        fp16_scales = (
            (
                layer.weight_scale.data.t().to(torch.float32)
                * layer.weight_global_scale.to(torch.float32)
            )
            .to(torch.float16)
            .contiguous()
        )
        packed_weight, packed_scales = sm70_ops.nvfp4_qpn4_prepare_sm70(
            qweight, fp16_scales
        )
    _store_state(
        layer,
        packed_weight,
        packed_scales,
        None,
        NVFP4_GROUP_SIZE,
        qweight.size(1),
        "nvfp4_qpn4",
        gated_silu=gated_silu,
        dense_weight_ptr=workspace.data_ptr(),
        global_scale=global_scale,
        use_scale_code=use_scale_code,
    )


def _wna16_dense_prefill_mm_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
    output_size: int,
    kernel_output_size: int,
    min_m: int,
) -> torch.Tensor:
    """Body of ``sm70_tm::wna16_dense_prefill_mm``; see the registered op."""
    from vllm import _sm70_ops as sm70_ops

    workspace = None
    if x.dtype == torch.float16:
        device_index = x.device.index
        if device_index is None:
            device_index = torch.accelerator.current_device_index()
        workspace = _wna16_prefill_dense_workspaces.get((device_index, torch.float16))
    if workspace is not None and x.shape[0] >= min_m:
        # Exact-dense prefill: expand the shared TurboMind uint4 encoding
        # into the bounded workspace and run cuBLAS. The dequant operator is
        # the AWQ one because both prepares share one packed encoding, one
        # GEMM consumer.
        k = x.shape[1]
        dense_weight = workspace[: k * output_size].view(k, output_size)
        sm70_ops.awq_sm70_dequantize_out(dense_weight, weight, scales, group_size)
        out = torch.mm(x, dense_weight)
        if kernel_output_size != output_size:
            padded = x.new_empty((x.shape[0], kernel_output_size))
            padded[:, :output_size] = out
            return padded
        return out
    out = x.new_empty((x.shape[0], kernel_output_size))
    sm70_ops.awq_gemm_sm70_out(out, x, weight, scales, group_size, k_ld, q_ld)
    return out


@torch.library.custom_op("sm70_tm::wna16_dense_prefill_mm", mutates_args=())
def wna16_dense_prefill_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
    output_size: int,
    kernel_output_size: int,
    min_m: int,
) -> torch.Tensor:
    """Route WNA16 uint4 linears between dense prefill and TurboMind.

    The M-dependent choice must stay inside one registered op. A Python
    branch on the symbolic token count compiles a shape guard into every
    compiled piece and pushed the captured decode step off its full-graph
    fast path (measured 3.4x decode-step regression on the production
    endpoint). The workspace is resolved from the module cache instead of
    the op schema, so the op stays a pure function of its inputs with no
    mutated arguments for dynamo and inductor to functionalize.
    """
    return _wna16_dense_prefill_mm_impl(
        x,
        weight,
        scales,
        group_size,
        k_ld,
        q_ld,
        output_size,
        kernel_output_size,
        min_m,
    )


@wna16_dense_prefill_mm.register_fake
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
    output_size: int,
    kernel_output_size: int,
    min_m: int,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], kernel_output_size))


def apply_prepared_linear(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    state = getattr(layer, STATE_ATTR)
    reshaped_x = x.reshape(-1, x.shape[-1])
    out_shape = x.shape[:-1] + (state.output_size,)
    kernel_output_size = state.padded_output_size or state.output_size
    if state.op_kind == "uint4" and state.prefill_dense_workspace is not None:
        # Exact-dense prefill route. The dense-vs-TurboMind choice lives
        # inside the registered op, opaque to dynamo; the eligibility check
        # here consults load-time state only, never the runtime shape.
        out = torch.ops.sm70_tm.wna16_dense_prefill_mm(
            reshaped_x,
            state.weight,
            state.scales,
            state.group_size,
            state.k_ld,
            state.q_ld,
            state.output_size,
            kernel_output_size,
            _SM70_WNA16_PREFILL_DENSE_MIN_M,
        )
    else:
        out = torch.empty(
            (reshaped_x.shape[0], kernel_output_size),
            dtype=x.dtype,
            device=x.device,
        )
        from vllm import _sm70_ops as sm70_ops

        if state.op_kind == "uint4":
            sm70_ops.awq_gemm_sm70_out(
                out,
                reshaped_x,
                state.weight,
                state.scales,
                state.group_size,
                state.k_ld,
                state.q_ld,
            )
        elif state.op_kind == "mxfp4":
            sm70_ops.mxfp4_gemm_sm70_out(
                out,
                reshaped_x,
                state.weight,
                state.scales,
                state.group_size,
                state.k_ld,
                state.q_ld,
            )
        elif state.op_kind == "nvfp4" and state.use_scale_code:
            sm70_ops.nvfp4_qpn2_compact_tm_gemm_sm70_out(
                out,
                reshaped_x,
                state.weight,
                state.scales,
                state.global_scale,
                state.k_ld,
                state.q_ld,
            )
        elif state.op_kind == "nvfp4":
            sm70_ops.nvfp4_gemm_sm70_out(
                out,
                reshaped_x,
                state.weight,
                state.scales,
                state.group_size,
                state.k_ld,
                state.q_ld,
            )
        elif state.op_kind == "nvfp4_qpn4":
            if reshaped_x.dtype != torch.float16:
                raise RuntimeError(
                    f"SM70 NVFP4 QPN4 requires float16 activations, "
                    f"got {reshaped_x.dtype}."
                )
            if reshaped_x.stride(-1) != 1:
                reshaped_x = reshaped_x.contiguous()
            sm70_ops.nvfp4_qpn4_dispatch_sm70_out(
                out,
                state.dense_weight_ptr,
                reshaped_x,
                state.weight,
                state.scales,
                state.global_scale,
                state.use_scale_code,
                False,
            )
        else:
            raise AssertionError(f"unknown SM70 TurboMind op kind: {state.op_kind}")
    if kernel_output_size != state.output_size:
        out = out[:, : state.output_size]
    if state.gated_silu and state.op_kind == "nvfp4":
        out_features = state.output_size // 2
        out = (
            out.reshape(reshaped_x.shape[0], out_features, 2)
            .transpose(1, 2)
            .reshape(reshaped_x.shape[0], state.output_size)
        )
    if bias is not None:
        out.add_(bias)
    return out.reshape(out_shape)


def apply_prepared_fused_silu_and_mul(
    layer: torch.nn.Module,
    x: torch.Tensor,
) -> torch.Tensor | None:
    state = getattr(layer, STATE_ATTR, None)
    if (
        state is None
        or state.op_kind not in ("nvfp4", "nvfp4_qpn4")
        or not state.gated_silu
    ):
        return None
    if x.dtype != torch.float16:
        raise RuntimeError(
            "SM70 TurboMind NVFP4 gated-SiLU requires float16 activations, "
            f"got {x.dtype}."
        )

    reshaped_x = x.reshape(-1, x.shape[-1])
    if reshaped_x.stride(-1) != 1:
        reshaped_x = reshaped_x.contiguous()
    out_features = state.output_size // 2
    out = torch.empty(
        (reshaped_x.shape[0], out_features),
        dtype=x.dtype,
        device=x.device,
    )
    if reshaped_x.shape[0] == 0:
        return out.reshape(*x.shape[:-1], out_features)

    from vllm import _sm70_ops as sm70_ops

    if state.op_kind == "nvfp4_qpn4":
        sm70_ops.nvfp4_qpn4_dispatch_sm70_out(
            out,
            state.dense_weight_ptr,
            reshaped_x,
            state.weight,
            state.scales,
            state.global_scale,
            state.use_scale_code,
            True,
        )
    else:
        sm70_ops.nvfp4_gemm_sm70_out(
            out,
            reshaped_x,
            state.weight,
            state.scales,
            state.group_size,
            state.k_ld,
            state.q_ld,
            True,
        )
    return out.reshape(*x.shape[:-1], out_features)
