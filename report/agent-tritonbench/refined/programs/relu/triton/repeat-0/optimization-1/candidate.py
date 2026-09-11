import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask)
    result = tl.where(values > 0, values, 0.0)
    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def wrapper(input, inplace=False):
        output = input if inplace else torch.empty_like(input)
        n_elements = input.numel()
        relu_kernel[(triton.cdiv(n_elements, 1024),)](
            input,
            output,
            n_elements,
            BLOCK=1024,
            num_warps=4,
        )
        return output

    return wrapper
