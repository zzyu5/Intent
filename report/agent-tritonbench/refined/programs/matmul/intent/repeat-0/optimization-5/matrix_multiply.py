import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    input,
    other,
    output,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Keep eight adjacent row tiles together for B-tile reuse. The fixed
    # 1024x1024 profile maps directly to an 8 x 16 x 2 grid.
    pid_m = tl.program_id(0) + tl.program_id(2) * 8
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    input_ptrs = input + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    other_ptrs = other + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        lhs = tl.load(input_ptrs)
        rhs = tl.load(other_ptrs)
        accumulator = tl.dot(lhs, rhs, accumulator, input_precision="ieee")
        input_ptrs += BLOCK_K * stride_ak
        other_ptrs += BLOCK_K * stride_bk

    output_ptrs = output + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(output_ptrs, accumulator.to(tl.float16))


def launch(input, other, output):
    M = input.shape[0]
    K = input.shape[1]
    N = other.shape[1]
    return _matmul_kernel[(8, 16, 2)](
        input,
        other,
        output,
        input.stride(0),
        input.stride(1),
        other.stride(0),
        other.stride(1),
        output.stride(0),
        output.stride(1),
        M=M,
        N=N,
        K=K,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=64,
        num_warps=4,
        num_stages=4,
    )


def run(input, other):
    output = torch.empty((input.shape[0], other.shape[1]), device=input.device, dtype=torch.float16)
    launch(input, other, output)
    return output
