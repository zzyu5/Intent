import torch
import triton
import triton.language as tl


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
    _softmax_kernel[(input.shape[0],)](
        input,
        output,
        ROWS_PER_PROGRAM=1,
        BLOCK_SIZE=1024,
        num_warps=8,
        num_stages=1,
    )


def run(input):
    output = torch.empty_like(input)
    launch(input, output)
    return output
