import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_1024(
    input_ptr,
    other_ptr,
    output_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = 1024 // BLOCK_M
    num_pid_n = 1024 // BLOCK_N

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    input_ptrs = input_ptr + offs_m[:, None] * 1024 + offs_k[None, :]
    other_ptrs = other_ptr + offs_k[:, None] * 1024 + offs_n[None, :]
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, 1024, BLOCK_K):
        input_tile = tl.load(input_ptrs)
        other_tile = tl.load(other_ptrs)
        accumulator = tl.dot(input_tile, other_tile, accumulator, out_dtype=tl.float32)
        input_ptrs += BLOCK_K
        other_ptrs += BLOCK_K * 1024

    output_ptrs = output_ptr + offs_m[:, None] * 1024 + offs_n[None, :]
    tl.store(output_ptrs, accumulator.to(tl.float16))


@triton.jit
def _matmul_strided(
    input_ptr,
    other_ptr,
    output_ptr,
    m,
    k,
    n,
    stride_input_m,
    stride_input_k,
    stride_other_k,
    stride_other_n,
    stride_output_m,
    stride_output_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(n, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    input_mask_m = offs_m < m
    other_mask_n = offs_n < n

    input_ptrs = input_ptr + offs_m[:, None] * stride_input_m + offs_k[None, :] * stride_input_k
    other_ptrs = other_ptr + offs_k[:, None] * stride_other_k + offs_n[None, :] * stride_other_n
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, k, BLOCK_K):
        k_offsets = k_start + offs_k
        k_mask = k_offsets < k
        input_tile = tl.load(input_ptrs, mask=input_mask_m[:, None] & k_mask[None, :], other=0.0)
        other_tile = tl.load(other_ptrs, mask=k_mask[:, None] & other_mask_n[None, :], other=0.0)
        accumulator = tl.dot(input_tile, other_tile, accumulator, out_dtype=tl.float32)
        input_ptrs += BLOCK_K * stride_input_k
        other_ptrs += BLOCK_K * stride_other_k

    output_ptrs = output_ptr + offs_m[:, None] * stride_output_m + offs_n[None, :] * stride_output_n
    tl.store(output_ptrs, accumulator.to(tl.float16), mask=input_mask_m[:, None] & other_mask_n[None, :])


def launch(input, other, output):
    if (
        input.shape == (1024, 1024)
        and other.shape == (1024, 1024)
        and input.dtype == torch.float16
        and other.dtype == torch.float16
        and input.is_contiguous()
        and other.is_contiguous()
        and output.shape == (1024, 1024)
        and output.dtype == torch.float16
        and output.is_contiguous()
    ):
        _matmul_1024[(16 * 16,)](
            input,
            other,
            output,
            BLOCK_M=64,
            BLOCK_N=64,
            BLOCK_K=64,
            GROUP_SIZE_M=8,
            num_warps=4,
            num_stages=3,
        )
        return

    m = input.shape[0]
    k = input.shape[1]
    n = other.shape[1]
    grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(n, META["BLOCK_N"]),)
    _matmul_strided[grid](
        input,
        other,
        output,
        m,
        k,
        n,
        input.stride(0),
        input.stride(1),
        other.stride(0),
        other.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=32,
    )


def run(input, other):
    output = torch.empty((input.shape[0], other.shape[1]), device=input.device, dtype=torch.float16)
    launch(input, other, output)
    return output
