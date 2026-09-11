import torch
import triton
import triton.language as tl


@triton.jit
def relu_fixed_kernel(input_ptr, output_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(input_ptr + offsets)
    tl.store(output_ptr + offsets, tl.maximum(values, 0.0))


@triton.jit
def relu_masked_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, tl.maximum(values, 0.0), mask=mask)


def build(context):
    def wrapper(input, inplace=False):
        output = input if inplace else torch.empty_like(input)
        n_elements = input.numel()
        if n_elements == 1048576:
            relu_fixed_kernel[(256,)](input, output, BLOCK=4096, num_warps=8)
        else:
            relu_masked_kernel[(triton.cdiv(n_elements, 4096),)](
                input, output, n_elements, BLOCK=4096, num_warps=8
            )
        return output

    return wrapper
