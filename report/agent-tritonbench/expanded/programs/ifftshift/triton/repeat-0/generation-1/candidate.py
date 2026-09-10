import torch
import triton
import triton.language as tl


@triton.jit
def _ifftshift_1d(input_ptr, output_ptr, n_elements, half, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    source = offsets + half
    source = tl.where(source < n_elements, source, source - n_elements)
    values = tl.load(input_ptr + source, mask=mask)
    tl.store(output_ptr + offsets, values, mask=mask)


def build(context):
    def wrapper(input, dim=None):
        output = torch.empty_like(input)
        n_elements = input.numel()
        _ifftshift_1d[(triton.cdiv(n_elements, 1024),)](
            input,
            output,
            n_elements,
            n_elements // 2,
            BLOCK=1024,
        )
        return output

    return wrapper
