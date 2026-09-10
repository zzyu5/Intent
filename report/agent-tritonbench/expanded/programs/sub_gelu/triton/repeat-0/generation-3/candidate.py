import torch
import triton
import triton.language as tl


@triton.jit
def _sub_gelu(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    input_value = tl.load(input_ptr + offsets, mask=mask)
    other_value = tl.load(other_ptr + offsets, mask=mask)
    value = input_value - other_value
    output_value = 0.5 * value * (1.0 + tl.erf(value * 0.7071067811865476))
    tl.store(output_ptr + offsets, output_value, mask=mask)


def build(context):
    def sub_gelu(input, other, alpha=1, approximate="none", out=None):
        output = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        _sub_gelu[(triton.cdiv(n_elements, 256),)](
            input, other, output, n_elements, BLOCK=256, num_warps=4
        )
        return output

    return sub_gelu
