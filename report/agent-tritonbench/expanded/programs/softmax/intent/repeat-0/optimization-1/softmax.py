import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
    ],
    key=["n_cols"],
)
@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    input_row = input_ptr + row * input_row_stride
    output_row = output_ptr + row * output_row_stride
    values = tl.load(input_row + offsets, mask=mask, other=-float("inf"))
    values = values.to(tl.float32)
    values = values - tl.max(values, axis=0)
    numerator = tl.exp(values)
    denominator = tl.sum(numerator, axis=0)
    tl.store(output_row + offsets, numerator / denominator, mask=mask)


def launch(input, output):
    n_rows, n_cols = input.shape
    grid = (n_rows,)
    _softmax_kernel[grid](
        input,
        output,
        n_cols,
        input.stride(0),
        output.stride(0),
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
