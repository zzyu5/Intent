import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 2048, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 2048, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 2048, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 2048, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 2048, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 2048, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 2048, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 2048, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 2048, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 2048, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 2048, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 2048, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 2048, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 2048, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 2048, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 2048, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _fused_rows(
    input,
    weight,
    bias,
    output,
    eps,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
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

    activated = tl.maximum(acc, 0.0, propagate_nan=tl.PropagateNan.ALL)
    denominator_n = tl.cast(N, tl.float32)
    mean = tl.fdiv(tl.sum(activated, axis=1), denominator_n, ieee_rounding=True)
    centered = activated - mean[:, None]
    variance = tl.fdiv(tl.sum(centered * centered, axis=1), denominator_n, ieee_rounding=True)
    denominator = libdevice.sqrt(variance + eps)
    normalized = tl.fdiv(centered, denominator[:, None], ieee_rounding=True)

    output_ptrs = output + offs_m[:, None] * N + offs_n[None, :]
    output_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(output_ptrs, normalized, mask=output_mask)


def launch(input, weight, bias, output, eps):
    m = input.shape[0]
    n = weight.shape[0]
    k = input.shape[1]
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]),)
    return _fused_rows[grid](input, weight, bias, output, eps, M=m, N=n, K=k)


def run(input, weight, bias, eps):
    output = torch.empty((input.shape[0], weight.shape[0]), device=input.device, dtype=torch.float32)
    launch(input, weight, bias, output, eps)
    return output
