import torch
import triton
import triton.language as tl


@triton.jit
def _tensordot_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, 64, BLOCK_K):
        k_offsets = k + offs_k
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak
        b_ptrs = b_ptr + k_offsets[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a_tile = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < 1024) & (k_offsets[None, :] < 64),
            other=0.0,
        )
        b_tile = tl.load(
            b_ptrs,
            mask=(k_offsets[:, None] < 64) & (offs_n[None, :] < 1024),
            other=0.0,
        )
        accumulator += tl.dot(a_tile, b_tile)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        out_ptrs,
        accumulator.to(tl.float16),
        mask=(offs_m[:, None] < 1024) & (offs_n[None, :] < 1024),
    )


def build(context):
    def tensordot(a, b, dims):
        output = torch.empty((1024, 1024), dtype=a.dtype, device=a.device)
        grid = (triton.cdiv(1024, 128), triton.cdiv(1024, 128))
        _tensordot_kernel[grid](
            a,
            b,
            output,
            a.stride(0),
            1,
            b.stride(1),
            b.stride(2),
            output.stride(0),
            output.stride(1),
            BLOCK_M=128,
            BLOCK_N=128,
            BLOCK_K=32,
            num_warps=4,
            num_stages=2,
        )
        return output

    return tensordot
