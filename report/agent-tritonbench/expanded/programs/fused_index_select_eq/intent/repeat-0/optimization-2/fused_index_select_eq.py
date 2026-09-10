import torch
import triton
import triton.language as tl


@triton.jit
def _fused_contiguous_kernel(
    input_ptr,
    index_ptr,
    output_ptr,
    other,
    BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_COLS)
    selected_row = tl.load(index_ptr + row).to(tl.int32)
    values = tl.load(input_ptr + selected_row * BLOCK_COLS + cols)
    result = (values == other).to(tl.int8)
    tl.store(output_ptr + row * BLOCK_COLS + cols, result)


@triton.jit
def _fused_strided_kernel(
    input_ptr,
    index_ptr,
    output_ptr,
    D1,
    D2,
    input_stride0,
    input_stride1,
    output_stride0,
    output_stride1,
    other,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_COLS)

    row_mask = rows < D1
    col_mask = cols < D2
    selected_rows = tl.load(index_ptr + rows, mask=row_mask, other=0)
    offsets = selected_rows[:, None] * input_stride0 + cols[None, :] * input_stride1
    values = tl.load(
        input_ptr + offsets,
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    result = (values == other).to(tl.int8)
    output_offsets = rows[:, None] * output_stride0 + cols[None, :] * output_stride1
    tl.store(
        output_ptr + output_offsets,
        result,
        mask=row_mask[:, None] & col_mask[None, :],
    )


def launch(input, index, output, other):
    D1 = index.shape[0]
    D2 = input.shape[1]
    grid = lambda META: (triton.cdiv(D1, META["BLOCK_ROWS"]),)
    return _fused_strided_kernel[grid](
        input,
        index,
        output,
        D1,
        D2,
        input.stride(0),
        input.stride(1),
        output.stride(0),
        output.stride(1),
        other,
        BLOCK_ROWS=1,
        BLOCK_COLS=128,
        num_warps=4,
        num_stages=2,
    )


def run(input, index, other):
    output = torch.empty((index.shape[0], input.shape[1]), device=input.device, dtype=torch.int8)
    _fused_contiguous_kernel[(index.shape[0],)](
        input,
        index,
        output,
        other,
        BLOCK_COLS=128,
        num_warps=4,
        num_stages=2,
    )
    return output
