import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(input_ptr, output_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(input_ptr + offsets)
    tl.store(output_ptr + offsets, tl.maximum(values, 0.0))


def build(context):
    def wrapper(input, inplace=False):
        output = input if inplace else torch.empty_like(input)
        relu_kernel[(128,)](
            input,
            output,
            BLOCK=8192,
            num_warps=8,
        )
        return output

    return wrapper
