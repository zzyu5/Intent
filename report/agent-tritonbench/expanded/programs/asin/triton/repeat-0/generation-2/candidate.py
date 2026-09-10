import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def asin_kernel(input_ptr, output_ptr, n_elements, input_stride, output_stride, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets * input_stride, mask=mask, other=0.0)
    values = libdevice.asin(values.to(tl.float32))
    tl.store(output_ptr + offsets * output_stride, values.to(tl.float16), mask=mask)


def build(context):
    def wrapper(input, *, out=None):
        output = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        asin_kernel[(triton.cdiv(n_elements, 256),)](
            input,
            output,
            n_elements,
            input.stride(0),
            output.stride(0),
            BLOCK=256,
        )
        return output

    return wrapper
