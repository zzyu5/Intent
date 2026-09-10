import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def tanh_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask)
    magnitude = libdevice.abs(values)
    exponential = libdevice.fast_expf(-2.0 * magnitude)
    result = libdevice.fast_dividef(1.0 - exponential, 1.0 + exponential)
    result = tl.where(values < 0.0, -result, result)
    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def wrapper(input, *, out=None):
        output = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        tanh_kernel[(triton.cdiv(n_elements, 2048),)](
            input,
            output,
            n_elements,
            BLOCK=2048,
            num_warps=8,
        )
        return output

    return wrapper
