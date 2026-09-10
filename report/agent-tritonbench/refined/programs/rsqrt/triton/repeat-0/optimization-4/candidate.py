import torch
import triton
import triton.language as tl


@triton.jit
def rsqrt_kernel(input_ptr, output_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(input_ptr + offsets)
    tl.store(output_ptr + offsets, tl.rsqrt(values))


def build(context):
    def wrapper(input, *, out=None):
        output = torch.empty_like(input) if out is None else out
        rsqrt_kernel[(2048,)](input, output, BLOCK=512, num_warps=2)
        return output

    return wrapper
