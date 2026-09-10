import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_ROWS": 1, "BLOCK_COLS": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_ROWS": 2, "BLOCK_COLS": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_ROWS": 4, "BLOCK_COLS": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_ROWS": 8, "BLOCK_COLS": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_ROWS": 1, "BLOCK_COLS": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_ROWS": 2, "BLOCK_COLS": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_ROWS": 4, "BLOCK_COLS": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_ROWS": 8, "BLOCK_COLS": 128}, num_warps=8, num_stages=1),
    ],
    key=["D1", "D2"],
)
@triton.jit
def _fused_kernel(
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

    input_offsets = selected_rows[:, None] * input_stride0 + cols[None, :] * input_stride1
    values = tl.load(
        input_ptr + input_offsets,
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    result = tl.cast(values == other, tl.int8)

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
    return _fused_kernel[grid](
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
    )


def run(input, index, other):
    output = torch.empty((index.shape[0], input.shape[1]), device=input.device, dtype=torch.int8)
    launch(input, index, output, other)
    return output
