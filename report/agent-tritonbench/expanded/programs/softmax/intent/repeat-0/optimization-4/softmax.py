import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"ROWS_PER_PROGRAM": 1}, num_warps=1, num_stages=1),
        triton.Config({"ROWS_PER_PROGRAM": 1}, num_warps=2, num_stages=1),
        triton.Config({"ROWS_PER_PROGRAM": 1}, num_warps=4, num_stages=1),
        triton.Config({"ROWS_PER_PROGRAM": 1}, num_warps=8, num_stages=1),
        triton.Config({"ROWS_PER_PROGRAM": 2}, num_warps=2, num_stages=1),
        triton.Config({"ROWS_PER_PROGRAM": 2}, num_warps=4, num_stages=1),
        triton.Config({"ROWS_PER_PROGRAM": 2}, num_warps=8, num_stages=1),
        triton.Config({"ROWS_PER_PROGRAM": 4}, num_warps=4, num_stages=1),
        triton.Config({"ROWS_PER_PROGRAM": 4}, num_warps=8, num_stages=1),
        triton.Config({"ROWS_PER_PROGRAM": 8}, num_warps=8, num_stages=1),
    ],
    key=[],
)
@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0) * ROWS_PER_PROGRAM
    row_offsets = row + tl.arange(0, ROWS_PER_PROGRAM)[:, None]
    col_offsets = tl.arange(0, BLOCK_SIZE)[None, :]
    offsets = row_offsets * BLOCK_SIZE + col_offsets
    values = tl.load(input_ptr + offsets).to(tl.float32)
    values = values - tl.max(values, axis=1)[:, None]
    numerator = tl.exp2(values * 1.4426950408889634)
    denominator = tl.sum(numerator, axis=1)[:, None]
    tl.store(output_ptr + offsets, numerator / denominator)


def launch(input, output):
    grid = lambda META: (triton.cdiv(input.shape[0], META["ROWS_PER_PROGRAM"]),)
    _softmax_kernel[grid](
        input,
        output,
        BLOCK_SIZE=1024,
    )


def run(input):
    output = torch.empty(
        (input.shape[0], input.shape[1]),
        device=input.device,
        dtype=torch.float32,
    )
    launch(input, output)
    return output
