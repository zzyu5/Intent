import torch
import triton
import triton.language as tl


@triton.jit
def exp_kernel(input_ptr, output_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(input_ptr + offsets)
    tl.store(output_ptr + offsets, tl.exp2(values * 1.4426950408889634))


def build(context):
    def wrapper(input, *, out=None):
        output = torch.empty_like(input) if out is None else out
        exp_kernel[(1024,)](
            input,
            output,
            BLOCK=1024,
            num_warps=8,
        )
        return output

    return wrapper
