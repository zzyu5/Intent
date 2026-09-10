import torch
import triton
import triton.language as tl


@triton.jit
def tanh_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask)
    result = 2.0 / (1.0 + tl.exp(-2.0 * values)) - 1.0
    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def wrapper(input, *, out=None):
        output = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        tanh_kernel[(triton.cdiv(n_elements, 256),)](
            input,
            output,
            n_elements,
            BLOCK=256,
        )
        return output

    return wrapper
