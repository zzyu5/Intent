import torch
import triton
import triton.language as tl


@triton.jit
def addmm_1024_kernel(
    input_ptr,
    mat1_ptr,
    mat2_ptr,
    out_ptr,
    out_stride_m,
    out_stride_n,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = 1024 // BLOCK_M
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, 1024, BLOCK_K):
        a = tl.load(mat1_ptr + offs_m[:, None] * 1024 + (k_start + offs_k)[None, :])
        b = tl.load(mat2_ptr + (k_start + offs_k)[:, None] * 1024 + offs_n[None, :])
        acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)

    if ALPHA != 1:
        acc *= ALPHA
    if BETA != 0:
        addend = tl.load(input_ptr + offs_m[:, None] * 1024 + offs_n[None, :])
        acc += addend.to(tl.float32) * BETA

    tl.store(
        out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n,
        acc,
    )


@triton.jit
def addmm_kernel(
    input_ptr,
    mat1_ptr,
    mat2_ptr,
    out_ptr,
    m,
    n,
    k,
    input_stride_m,
    input_stride_n,
    mat1_stride_m,
    mat1_stride_k,
    mat2_stride_k,
    mat2_stride_n,
    out_stride_m,
    out_stride_n,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, k, BLOCK_K):
        k_offsets = k_start + offs_k
        a = tl.load(
            mat1_ptr
            + offs_m[:, None] * mat1_stride_m
            + k_offsets[None, :] * mat1_stride_k,
            mask=(offs_m[:, None] < m) & (k_offsets[None, :] < k),
            other=0.0,
        )
        b = tl.load(
            mat2_ptr
            + k_offsets[:, None] * mat2_stride_k
            + offs_n[None, :] * mat2_stride_n,
            mask=(k_offsets[:, None] < k) & (offs_n[None, :] < n),
            other=0.0,
        )
        acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)

    if ALPHA != 1:
        acc *= ALPHA
    if BETA != 0:
        addend = tl.load(
            input_ptr
            + offs_m[:, None] * input_stride_m
            + offs_n[None, :] * input_stride_n,
            mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
            other=0.0,
        )
        acc += addend.to(tl.float32) * BETA

    tl.store(
        out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )


def build(context):
    def wrapper(input, mat1, mat2, *, beta=1, alpha=1, out=None):
        m = mat1.shape[0]
        k = mat1.shape[1]
        n = mat2.shape[1]
        result = out
        if result is None:
            result = torch.empty((m, n), device=mat1.device, dtype=mat1.dtype)

        fixed_contiguous = (
            m == 1024
            and n == 1024
            and k == 1024
            and input.shape == (1024, 1024)
            and input.dtype == torch.float16
            and mat1.dtype == torch.float16
            and mat2.dtype == torch.float16
            and result.dtype == torch.float16
            and input.stride(0) == 1024
            and input.stride(1) == 1
            and mat1.stride(0) == 1024
            and mat1.stride(1) == 1
            and mat2.stride(0) == 1024
            and mat2.stride(1) == 1
            and result.shape == (1024, 1024)
        )

        if fixed_contiguous:
            addmm_1024_kernel[(1024 // 128) * (1024 // 256),](
                input,
                mat1,
                mat2,
                result,
                result.stride(0),
                result.stride(1),
                ALPHA=alpha,
                BETA=beta,
                BLOCK_M=128,
                BLOCK_N=256,
                BLOCK_K=64,
                num_warps=8,
                num_stages=3,
            )
            return result

        addmm_kernel[(triton.cdiv(m, 128), triton.cdiv(n, 128))](
            input,
            mat1,
            mat2,
            result,
            m,
            n,
            k,
            input.stride(0),
            input.stride(1),
            mat1.stride(0),
            mat1.stride(1),
            mat2.stride(0),
            mat2.stride(1),
            result.stride(0),
            result.stride(1),
            ALPHA=alpha,
            BETA=beta,
            BLOCK_M=128,
            BLOCK_N=128,
            BLOCK_K=32,
            num_warps=8,
            num_stages=4,
        )
        return result

    return wrapper
