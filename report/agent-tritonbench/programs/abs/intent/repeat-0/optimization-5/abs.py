import torch
import triton
import triton.language as tl


@triton.jit
def abs_kernel(input, output, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(input + offsets)
    tl.store(output + offsets, tl.abs(value))


def launch(input, output):
    return abs_kernel[(1024,)](input, output, BLOCK=1024, num_warps=1)


def run(input):
    output = torch.empty((input.shape[0],), device=input.device, dtype=input.dtype)
    launch(input, output)
    return output
