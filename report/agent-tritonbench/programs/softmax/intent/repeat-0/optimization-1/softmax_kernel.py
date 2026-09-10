import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=1),
    ],
    key=["n_rows", "n_cols"],
)
@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    input_row = input_ptr + row * input_row_stride
    output_row = output_ptr + row * output_row_stride

    values = tl.load(input_row + offsets, mask=mask, other=-float("inf"))
    row_max = tl.max(values, axis=0)
    exponent = tl.exp2((values - row_max) * 1.4426950408889634)
    exponent = tl.where(mask, exponent, 0.0)
    denominator = tl.sum(exponent, axis=0)
    result = exponent / denominator
    tl.store(output_row + offsets, result, mask=mask)


def launch(input, output):
    n_rows = input.shape[0]
    n_cols = input.shape[1]
    _softmax_kernel[(n_rows,)](
        input,
        output,
        input.stride(0),
        output.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=1024,
    )


def run(input):
    output = torch.empty((input.shape[0], input.shape[1]), device=input.device, dtype=torch.float32)
    launch(input, output)
    return output
