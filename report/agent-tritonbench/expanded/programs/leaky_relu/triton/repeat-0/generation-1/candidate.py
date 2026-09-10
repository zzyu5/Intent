import torch
import triton
import triton.language as tl


@triton.jit
def leaky_relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    negative_slope,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    result = tl.where(values >= 0, values, values * negative_slope)
    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def wrapper(input, negative_slope=0.01, inplace=False):
        output = input if inplace else torch.empty_like(input)
        n_elements = input.numel()
        if n_elements:
            leaky_relu_kernel[(triton.cdiv(n_elements, 1024),)](
                input,
                output,
                n_elements,
                negative_slope,
                BLOCK=1024,
                num_warps=4,
            )
        return output

    return wrapper
