import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


_LINEAR_CONFIGS = [
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
]


@triton.autotune(configs=_LINEAR_CONFIGS, key=["M", "N", "K"])
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
        a = tl.load(
            input + offs_m[:, None] * K + offs_k[None, :],
            mask=row_mask[:, None] & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            weight + offs_n[:, None] * K + offs_k[None, :],
            mask=col_mask[:, None] & (offs_k[None, :] < K),
            other=0.0,
        )
        acc = tl.dot(a, tl.permute(b, (1, 0)), acc, input_precision="ieee")

    acc = tl.maximum(acc, 0.0, propagate_nan=tl.PropagateNan.ALL)
    tl.store(
        output + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@triton.autotune(configs=_LINEAR_CONFIGS, key=["M", "N", "K"])
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
        a = tl.load(
            input + offs_m[:, None] * K + offs_k[None, :],
            mask=row_mask[:, None] & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            weight + offs_n[:, None] * K + offs_k[None, :],
            mask=col_mask[:, None] & (offs_k[None, :] < K),
            other=0.0,
        )
        acc = tl.dot(a, tl.permute(b, (1, 0)), acc, input_precision="ieee")

    acc = tl.maximum(acc, 0.0, propagate_nan=tl.PropagateNan.ALL)
    tl.store(
        output + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 2048}, num_warps=16, num_stages=2),
    ],
    key=["N"],
)
@triton.jit
def _layer_norm_inplace(
    output,
    eps,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    values = tl.load(output + row * N + offsets, mask=mask, other=0.0)
    denominator_n = tl.cast(N, tl.float32)
    mean = tl.fdiv(tl.sum(values, axis=0), denominator_n, ieee_rounding=True)
    centered = values - mean
    squared = tl.where(mask, centered * centered, 0.0)
    variance = tl.fdiv(tl.sum(squared, axis=0), denominator_n, ieee_rounding=True)
    denominator = libdevice.sqrt(variance + eps)
    normalized = tl.fdiv(centered, denominator, ieee_rounding=True)
    tl.store(output + row * N + offsets, normalized, mask=mask)


def launch(input, weight, bias, output, eps):
    m = input.shape[0]
    n = weight.shape[0]
    k = input.shape[1]
    linear_grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]),
        triton.cdiv(n, meta["BLOCK_N"]),
    )
    if bias is None:
        _linear_relu_no_bias[linear_grid](
            input, weight, output, M=m, N=n, K=k
        )
    else:
        _linear_relu_bias[linear_grid](
            input, weight, bias, output, M=m, N=n, K=k
        )
    _layer_norm_inplace[(m,)](output, eps, M=m, N=n)


def run(input, weight, bias, eps):
    output = torch.empty(
        (input.shape[0], weight.shape[0]),
        device=input.device,
        dtype=torch.float32,
    )
    launch(input, weight, bias, output, eps)
    return output
