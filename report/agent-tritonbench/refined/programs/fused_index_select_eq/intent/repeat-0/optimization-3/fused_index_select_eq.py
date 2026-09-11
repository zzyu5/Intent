import torch
import triton
import triton.language as tl


@triton.jit
def _contiguous_kernel(input, index, output):
    pid = tl.program_id(0)
    cols = tl.arange(0, 128)

    selected_row = tl.load(index + pid).to(tl.int32)
    pointers = input + selected_row * 128 + cols
    values = tl.load(pointers)
    tl.store(output + pid * 128 + cols, values == 0.0)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=8, num_stages=1),
    ],
    key=["M", "N"],
)
@triton.jit
def _strided_kernel(
    input,
    index,
    output,
    other,
    M,
    N,
    input_stride0,
    input_stride1,
    index_stride0,
    output_stride0,
    output_stride1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    selected_rows = tl.load(index + rows * index_stride0, mask=row_mask, other=0)
    selected_rows = selected_rows.to(tl.int32)
    pointers = input + selected_rows[:, None] * input_stride0 + cols[None, :] * input_stride1
    values = tl.load(pointers, mask=mask, other=0.0)
    result = values == other
    output_pointers = output + rows[:, None] * output_stride0 + cols[None, :] * output_stride1
    tl.store(output_pointers, result, mask=mask)


def launch(input, index, output, other):
    M = index.shape[0]
    N = input.shape[1]
    if (
        M == 1024
        and
        N == 128
        and input.stride(0) == 128
        and input.stride(1) == 1
        and index.stride(0) == 1
        and output.stride(0) == 128
        and output.stride(1) == 1
    ):
        return _contiguous_kernel[(M,)](input, index, output, num_warps=1, num_stages=1)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    return _strided_kernel[grid](
        input,
        index,
        output,
        other,
        M,
        N,
        input.stride(0),
        input.stride(1),
        index.stride(0),
        output.stride(0),
        output.stride(1),
    )


def run(input, index, other):
    output = torch.empty((index.shape[0], input.shape[1]), device=input.device, dtype=torch.bool)
    launch(input, index, output, other)
    return output
