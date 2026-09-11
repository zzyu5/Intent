import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _linear_relu_bias(
    input,
    weight,
    bias,
    output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = offs_m < M
    col_mask = offs_n < N
    bias_value = tl.load(bias + offs_n, mask=col_mask, other=0.0)
    acc = tl.broadcast_to(bias_value[None, :], (BLOCK_M, BLOCK_N))

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        input_ptrs = input + offs_m[:, None] * K + offs_k[None, :]
        input_mask = row_mask[:, None] & (offs_k[None, :] < K)
        a = tl.load(input_ptrs, mask=input_mask, other=0.0)

        weight_ptrs = weight + offs_n[:, None] * K + offs_k[None, :]
        weight_mask = col_mask[:, None] & (offs_k[None, :] < K)
        b = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
        b = tl.permute(b, (1, 0))
        acc = tl.dot(a, b, acc, input_precision="ieee")

    acc = tl.maximum(acc, 0.0, propagate_nan=tl.PropagateNan.ALL)
    output_ptrs = output + offs_m[:, None] * N + offs_n[None, :]
    output_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(output_ptrs, acc, mask=output_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _linear_relu_no_bias(
    input,
    weight,
    output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = offs_m < M
    col_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        input_ptrs = input + offs_m[:, None] * K + offs_k[None, :]
        input_mask = row_mask[:, None] & (offs_k[None, :] < K)
        a = tl.load(input_ptrs, mask=input_mask, other=0.0)

        weight_ptrs = weight + offs_n[:, None] * K + offs_k[None, :]
        weight_mask = col_mask[:, None] & (offs_k[None, :] < K)
        b = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
        b = tl.permute(b, (1, 0))
        acc = tl.dot(a, b, acc, input_precision="ieee")

    acc = tl.maximum(acc, 0.0, propagate_nan=tl.PropagateNan.ALL)
    output_ptrs = output + offs_m[:, None] * N + offs_n[None, :]
    output_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(output_ptrs, acc, mask=output_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 2048}, num_warps=16, num_stages=2),
    ],
    key=["M", "N"],
)
@triton.jit
def _layer_norm(
    input,
    output,
    eps,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    values = tl.load(input + row * N + offsets, mask=mask, other=0.0)
    denominator_n = tl.cast(N, tl.float32)
    mean = tl.fdiv(tl.sum(values, axis=0), denominator_n, ieee_rounding=True)
    centered = values - mean
    squared = tl.where(mask, centered * centered, 0.0)
    variance = tl.fdiv(tl.sum(squared, axis=0), denominator_n, ieee_rounding=True)
    denominator = libdevice.sqrt(variance + eps)
    normalized = tl.fdiv(centered, denominator, ieee_rounding=True)
    tl.store(output + row * N + offsets, normalized, mask=mask)


def _launch_linear(input, weight, bias, output):
    M = input.shape[0]
    K = input.shape[1]
    N = weight.shape[0]
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    if bias is None:
        _linear_relu_no_bias[grid](input, weight, output, M, N, K)
    else:
        _linear_relu_bias[grid](input, weight, bias, output, M, N, K)


def _launch_norm(input, output, eps):
    M = input.shape[0]
    N = input.shape[1]
    _layer_norm[(M,)](input, output, eps, M, N)


def launch(input, weight, bias, output, eps):
    activated = torch.empty_like(output)
    _launch_linear(input, weight, bias, activated)
    _launch_norm(activated, output, eps)


def run(input, weight, bias, eps):
    activated = torch.empty((input.shape[0], weight.shape[0]), device=input.device, dtype=torch.float32)
    output = torch.empty_like(activated)
    _launch_linear(input, weight, bias, activated)
    _launch_norm(activated, output, eps)
    return output
