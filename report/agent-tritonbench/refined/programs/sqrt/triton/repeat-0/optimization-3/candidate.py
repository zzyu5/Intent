import torch
import triton
import triton.language as tl


@triton.jit
def sqrt_kernel(input_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(input_ptr + offsets)
    result = tl.sqrt(values)
    tl.store(output_ptr + offsets, result)


def build(context):
    def wrapper(input, *, out=None):
        output = torch.empty_like(input) if out is None else out
        sqrt_kernel[(1024,)](input, output, BLOCK_SIZE=1024, num_warps=4)
        return output

    return wrapper
