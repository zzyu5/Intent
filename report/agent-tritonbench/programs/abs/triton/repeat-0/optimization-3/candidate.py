import torch
import triton
import triton.language as tl


@triton.jit
def _abs_kernel(input_ptr, output_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(input_ptr + offsets)
    tl.store(output_ptr + offsets, tl.abs(values))


def build(context):
    def wrapper(input, *, out=None):
        output = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        if n_elements:
            _abs_kernel[(triton.cdiv(n_elements, 2048),)](
                input,
                output,
                BLOCK=2048,
                num_warps=4,
                num_stages=1,
            )
        return output

    return wrapper
