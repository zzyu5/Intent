import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _fused_row(
    input,
    weight,
    bias,
    output,
    eps,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    bias_value = tl.load(bias + offs_n)
    acc = tl.broadcast_to(bias_value[None, :], (1, BLOCK_N))

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a = tl.load(input + row * K + offs_k)[None, :]
        b = tl.load(weight + offs_n[:, None] * K + offs_k[None, :])
        b = tl.permute(b, (1, 0))
        acc = tl.dot(a, b, acc, input_precision="ieee")

    acc = tl.maximum(acc, 0.0, propagate_nan=tl.PropagateNan.ALL)
    denominator_n = tl.cast(N, tl.float32)
    mean = tl.fdiv(tl.sum(acc, axis=1), denominator_n, ieee_rounding=True)
    centered = acc - mean[:, None]
    variance = tl.fdiv(tl.sum(centered * centered, axis=1), denominator_n, ieee_rounding=True)
    denominator = libdevice.sqrt(variance + eps)
    normalized = tl.fdiv(centered, denominator[:, None], ieee_rounding=True)
    tl.store(output + row * N + offs_n, normalized[0, :])


def run(input, weight, bias, eps):
    m = input.shape[0]
    n = weight.shape[0]
    k = input.shape[1]
    output = torch.empty((m, n), device=input.device, dtype=torch.float32)
    _fused_row[(m,)](
        input,
        weight,
        bias,
        output,
        eps,
        M=m,
        N=n,
        K=k,
        BLOCK_N=2048,
        BLOCK_K=64,
        num_warps=8,
        num_stages=2,
    )
    return output
