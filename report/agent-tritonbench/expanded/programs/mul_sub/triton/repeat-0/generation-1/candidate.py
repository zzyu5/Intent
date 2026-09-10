import torch
import triton
import triton.language as tl


@triton.jit
def mul_sub_kernel(
    input_ptr,
    other_mul_ptr,
    other_sub_ptr,
    output_ptr,
    n_elements,
    alpha,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    input_values = tl.load(input_ptr + offsets, mask=mask)
    mul_values = tl.load(other_mul_ptr + offsets, mask=mask)
    sub_values = tl.load(other_sub_ptr + offsets, mask=mask)
    result = input_values * mul_values - alpha * sub_values
    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def wrapper(input, other_mul, other_sub, alpha=1, out=None):
        output = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        mul_sub_kernel[(triton.cdiv(n_elements, 256),)](
            input,
            other_mul,
            other_sub,
            output,
            n_elements,
            alpha,
            BLOCK=256,
        )
        return output

    return wrapper
