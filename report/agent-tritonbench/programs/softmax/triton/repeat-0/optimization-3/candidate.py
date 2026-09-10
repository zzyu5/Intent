import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    program_id = tl.program_id(0)
    rows = program_id * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    cols = tl.arange(0, BLOCK_SIZE)
    offsets = rows[:, None] * BLOCK_SIZE + cols[None, :]

    values = tl.load(input_ptr + offsets)
    row_max = tl.max(values, axis=1)
    numerators = tl.exp(values - row_max[:, None])
    denominator = tl.sum(numerators, axis=1)
    result = numerators / denominator[:, None]
    tl.store(output_ptr + offsets, result)


def build(context):
    def wrapper(input, dim, dtype=None):
        if dim < 0:
            dim += input.ndim
        if dim != 1:
            raise ValueError("This implementation supports dim=1 only")

        output_dtype = input.dtype if dtype is None else dtype
        output = torch.empty(input.shape, device=input.device, dtype=output_dtype)
        n_rows = input.shape[0]
        rows_per_program = 2
        _softmax_kernel[(triton.cdiv(n_rows, rows_per_program),)](
            input,
            output,
            BLOCK_SIZE=1024,
            ROWS_PER_PROGRAM=rows_per_program,
            num_warps=4,
        )
        return output

    return wrapper
