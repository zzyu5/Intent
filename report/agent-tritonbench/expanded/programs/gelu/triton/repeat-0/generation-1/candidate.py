import torch
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    y = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
    tl.store(output_ptr + offsets, y, mask=mask)


def build(context):
    def wrapper(input, approximate="none"):
        output = torch.empty_like(input)
        n_elements = input.numel()
        gelu_kernel[(triton.cdiv(n_elements, 256),)](
            input, output, n_elements, BLOCK_SIZE=256
        )
        return output

    return wrapper
