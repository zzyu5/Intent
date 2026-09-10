import torch
import triton
import triton.language as tl


_FIXED_MATMUL_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=2, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=5),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=_FIXED_MATMUL_CONFIGS, key=[])
@triton.jit
def _contiguous_fixed_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = 1024 // BLOCK_M
    num_pid_n = 1024 // BLOCK_N

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, 1024, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * 1024 + (k_start + offs_k)[None, :])
        b = tl.load(b_ptr + (k_start + offs_k)[:, None] * 1024 + offs_n[None, :])
        accumulator = tl.dot(a, b, accumulator)

    tl.store(
        c_ptr + offs_m[:, None] * 1024 + offs_n[None, :],
        accumulator.to(tl.float16),
    )


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(m, BLOCK_M)
    num_pid_n = tl.cdiv(n, BLOCK_N)

    # Group neighboring rows so adjacent programs reuse the same B tiles in L2.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_mask = (offs_m[:, None] < m) & (offs_k[None, :] < k)
    b_mask = (offs_k[:, None] < k) & (offs_n[None, :] < n)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=a_mask,
            other=0.0,
        )
        b = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=b_mask,
            other=0.0,
        )
        accumulator = tl.dot(a, b, accumulator)
        offs_k += BLOCK_K
        a_mask = (offs_m[:, None] < m) & (offs_k[None, :] < k)
        b_mask = (offs_k[:, None] < k) & (offs_n[None, :] < n)

    c = accumulator.to(tl.float16)
    c_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=c_mask,
    )


def build(context):
    def wrapper(input, other, *, out=None):
        m, k = input.shape
        _, n = other.shape
        if out is None:
            out = torch.empty((m, n), device=input.device, dtype=input.dtype)

        if (
            input.shape == (1024, 1024)
            and other.shape == (1024, 1024)
            and input.dtype == torch.float16
            and other.dtype == torch.float16
            and out.dtype == torch.float16
            and input.stride() == (1024, 1)
            and other.stride() == (1024, 1)
            and out.stride() == (1024, 1)
        ):
            grid = lambda meta: (1024 // meta['BLOCK_M'] * (1024 // meta['BLOCK_N']),)
            _contiguous_fixed_matmul_kernel[grid](
                input,
                other,
                out,
            )
        else:
            grid = (triton.cdiv(m, 128) * triton.cdiv(n, 128),)
            _matmul_kernel[grid](
                input,
                other,
                out,
                m,
                n,
                k,
                input.stride(0),
                input.stride(1),
                other.stride(0),
                other.stride(1),
                out.stride(0),
                out.stride(1),
                BLOCK_M=128,
                BLOCK_N=128,
                BLOCK_K=32,
                GROUP_M=8,
                num_warps=8,
                num_stages=4,
            )
        return out

    return wrapper
