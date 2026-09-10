import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_rows(
    input_ptr,
    output_ptr,
    n_cols,
    input_stride,
    output_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    input_row = input_ptr + row * input_stride
    values = tl.load(input_row + offsets, mask=mask, other=-float("inf"))
    values = values.to(tl.float32)
    row_max = tl.max(values, axis=0)
    exponentials = tl.exp(values - row_max)
    denominator = tl.sum(exponentials, axis=0)
    probabilities = exponentials / denominator

    output_row = output_ptr + row * output_stride
    tl.store(output_row + offsets, probabilities, mask=mask)


def build(context):
    def wrapper(input, dim, dtype=None):
        output = torch.empty_like(input, dtype=dtype)
        rows = input.shape[0]
        cols = input.shape[1]
        _softmax_rows[(rows,)](
            input,
            output,
            cols,
            input.stride(0),
            output.stride(0),
            BLOCK_SIZE=1024,
            num_warps=8,
        )
        return output

    return wrapper
