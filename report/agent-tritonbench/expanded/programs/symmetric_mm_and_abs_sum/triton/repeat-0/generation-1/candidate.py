import torch
import triton
import triton.language as tl


@triton.jit
def _zero_scalar(out_ptr):
    tl.store(out_ptr, 0.0)


@triton.jit
def _symmetric_mm_abs_sum(
    a_ptr,
    c_ptr,
    out_ptr,
    n_rows,
    n_cols,
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    alpha,
    beta,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = offs_m < n_rows
    cols = offs_n < n_rows

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(n_cols, BLOCK_K)):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < n_cols

        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=rows[:, None] & k_mask[None, :],
            other=0.0,
        )
        a_transpose = tl.load(
            a_ptr + offs_k[:, None] * stride_ak + offs_n[None, :] * stride_am,
            mask=k_mask[:, None] & cols[None, :],
            other=0.0,
        )
        acc += tl.dot(a, a_transpose)

    c = tl.load(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        mask=rows[:, None] & cols[None, :],
        other=0.0,
    )
    result = alpha * acc + beta * c
    tile_sum = tl.sum(tl.sum(tl.abs(result), axis=1), axis=0)
    tl.atomic_add(out_ptr, tile_sum)


def build(context):
    del context

    def wrapper(A, C, alpha, beta):
        n_rows = A.shape[0]
        n_cols = A.shape[1]
        result = torch.empty((), device=A.device, dtype=A.dtype)

        _zero_scalar[(1,)](result)
        _symmetric_mm_abs_sum[(triton.cdiv(n_rows, 128), triton.cdiv(n_rows, 128))](
            A,
            C,
            result,
            n_rows,
            n_cols,
            A.stride(0),
            A.stride(1),
            C.stride(0),
            C.stride(1),
            alpha,
            beta,
            BLOCK_M=128,
            BLOCK_N=128,
            BLOCK_K=32,
            num_warps=8,
            num_stages=3,
        )
        return result

    return wrapper
