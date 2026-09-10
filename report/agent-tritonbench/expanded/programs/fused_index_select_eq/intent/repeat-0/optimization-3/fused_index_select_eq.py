import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_ROWS": 1, "BLOCK_COLS": 128}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_ROWS": 1, "BLOCK_COLS": 128}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_ROWS": 1, "BLOCK_COLS": 128}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_ROWS": 1, "BLOCK_COLS": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_ROWS": 1, "BLOCK_COLS": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_ROWS": 1, "BLOCK_COLS": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_ROWS": 1, "BLOCK_COLS": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_ROWS": 1, "BLOCK_COLS": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_ROWS": 2, "BLOCK_COLS": 128}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_ROWS": 2, "BLOCK_COLS": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_ROWS": 2, "BLOCK_COLS": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_ROWS": 2, "BLOCK_COLS": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_ROWS": 4, "BLOCK_COLS": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_ROWS": 4, "BLOCK_COLS": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_ROWS": 8, "BLOCK_COLS": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_ROWS": 8, "BLOCK_COLS": 128}, num_warps=8, num_stages=2),
    ],
    key=["n_rows"],
)
@triton.jit
def _fused_contiguous_kernel(
    input_ptr,
    index_ptr,
    output_ptr,
    n_rows,
    other,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_COLS)

    selected_rows = tl.load(index_ptr + rows).to(tl.int32)
    input_offsets = selected_rows[:, None] * BLOCK_COLS + cols[None, :]
    values = tl.load(input_ptr + input_offsets)
    result = (values == other).to(tl.int8)

    output_offsets = rows[:, None] * BLOCK_COLS + cols[None, :]
    tl.store(output_ptr + output_offsets, result)


def launch(input, index, output, other):
    n_rows = index.shape[0]
    grid = lambda META: (triton.cdiv(n_rows, META["BLOCK_ROWS"]),)
    return _fused_contiguous_kernel[grid](
        input,
        index,
        output,
        n_rows,
        other,
    )


def run(input, index, other):
    output = torch.empty((index.shape[0], input.shape[1]), device=input.device, dtype=torch.int8)
    launch(input, index, output, other)
    return output
