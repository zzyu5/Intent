import torch
import triton
import triton.language as tl


@triton.jit
def _gather_masked_fill(
    input_ptr,
    index_ptr,
    mask_ptr,
    output_ptr,
    value,
    n_rows,
    n_cols,
    input_stride0,
    input_stride1,
    index_stride0,
    index_stride1,
    mask_stride0,
    mask_stride1,
    output_stride0,
    output_stride1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)

    index_offsets = rows[:, None] * index_stride0 + cols[None, :] * index_stride1
    indices = tl.load(index_ptr + index_offsets, mask=valid, other=0)
    input_offsets = indices * input_stride0 + cols[None, :] * input_stride1
    gathered = tl.load(input_ptr + input_offsets, mask=valid, other=0.0)

    mask_offsets = rows[:, None] * mask_stride0 + cols[None, :] * mask_stride1
    selected = tl.load(mask_ptr + mask_offsets, mask=valid, other=0)
    result = tl.where(selected, value, gathered)

    output_offsets = rows[:, None] * output_stride0 + cols[None, :] * output_stride1
    tl.store(output_ptr + output_offsets, result, mask=valid)


def launch(input, index, mask, output, value):
    n_rows, n_cols = input.shape
    grid = (triton.cdiv(n_rows, 4), triton.cdiv(n_cols, 1024))
    _gather_masked_fill[grid](
        input,
        index,
        mask,
        output,
        value,
        n_rows,
        n_cols,
        input.stride(0),
        input.stride(1),
        index.stride(0),
        index.stride(1),
        mask.stride(0),
        mask.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=4,
        BLOCK_N=1024,
        num_warps=8,
        num_stages=2,
    )


def run(input, index, mask, value):
    output = torch.empty_like(input)
    launch(input, index, mask, output, value)
    return output
