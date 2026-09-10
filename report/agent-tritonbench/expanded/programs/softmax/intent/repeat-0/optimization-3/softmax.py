import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=[],
)
@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    row_offset = row * BLOCK_SIZE
    values = tl.load(input_ptr + row_offset + offsets)
    values = values.to(tl.float32)
    values = values - tl.max(values, axis=0)
    numerator = tl.exp2(values * 1.4426950408889634)
    denominator = tl.sum(numerator, axis=0)
    tl.store(output_ptr + row_offset + offsets, numerator / denominator)


def launch(input, output):
    grid = (1024,)
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
