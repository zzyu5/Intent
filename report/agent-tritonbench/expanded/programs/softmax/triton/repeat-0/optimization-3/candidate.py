import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_rows(
    input_ptr,
    output_ptr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    for row_offset in range(ROWS_PER_PROGRAM):
        row = program_id * ROWS_PER_PROGRAM + row_offset
        input_row = input_ptr + row * BLOCK_SIZE
        values = tl.load(input_row + offsets)
        values = values.to(tl.float32)
        row_max = tl.max(values, axis=0)
        exponentials = tl.exp(values - row_max)
        denominator = tl.sum(exponentials, axis=0)
        probabilities = exponentials / denominator

        output_row = output_ptr + row * BLOCK_SIZE
        tl.store(output_row + offsets, probabilities)


def build(context):
    def wrapper(input, dim, dtype=None):
        output = torch.empty_like(input, dtype=dtype)
        rows = input.shape[0]
        _softmax_rows[(triton.cdiv(rows, 2),)](
            input,
            output,
            ROWS_PER_PROGRAM=2,
            BLOCK_SIZE=1024,
            num_warps=4,
        )
        return output

    return wrapper
