import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def tanh_kernel(input_ptr, output_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(input_ptr + offsets)
    exponential = libdevice.fast_expf(-2.0 * values)
    result = libdevice.fast_dividef(2.0, 1.0 + exponential) - 1.0
    tl.store(output_ptr + offsets, result)


def build(context):
    def wrapper(input, *, out=None):
        output = torch.empty_like(input) if out is None else out
        tanh_kernel[(256,)](
            input,
            output,
            BLOCK=4096,
            num_warps=8,
        )
        return output

    return wrapper
