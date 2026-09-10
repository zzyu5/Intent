import torch
import triton
import triton.language as tl


@triton.jit
def mul_kernel(input_ptr, other_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    input_values = tl.load(input_ptr + offsets, mask=mask)
    other_values = tl.load(other_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, input_values * other_values, mask=mask)


def build(context):
    def wrapper(input, other, *, out=None):
        output = out if out is not None else torch.empty_like(input)
        count = input.numel()
        mul_kernel[(triton.cdiv(count, 1024),)](
            input,
            other,
            output,
            count,
            BLOCK=1024,
            num_warps=4,
        )
        return output

    return wrapper
