import math

import torch
import triton
import triton.language as tl


@triton.jit
def _add_gelu_tensor(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    alpha,
    ALPHA_IS_ONE: tl.constexpr,
    APPROXIMATE_TANH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    other = tl.load(other_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if ALPHA_IS_ONE:
        x = x + other
    else:
        x = x + alpha * other

    if APPROXIMATE_TANH:
        x3 = x * x * x
        result = 0.5 * x * (1.0 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x3)))
    else:
        result = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _add_gelu_scalar(
    input_ptr,
    other,
    output_ptr,
    n_elements,
    alpha,
    ALPHA_IS_ONE: tl.constexpr,
    APPROXIMATE_TANH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if ALPHA_IS_ONE:
        x = x + other
    else:
        x = x + alpha * other

    if APPROXIMATE_TANH:
        x3 = x * x * x
        result = 0.5 * x * (1.0 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x3)))
    else:
        result = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def wrapper(input, other, alpha=1, approximate="none", out=None):
        output = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        alpha_value = float(alpha)
        alpha_is_one = alpha_value == 1.0
        approximate_tanh = approximate == "tanh"

        if isinstance(other, torch.Tensor):
            _add_gelu_tensor[grid](
                input,
                other,
                output,
                n_elements,
                alpha_value,
                ALPHA_IS_ONE=alpha_is_one,
                APPROXIMATE_TANH=approximate_tanh,
                BLOCK_SIZE=1024,
                num_warps=4,
            )
        else:
            _add_gelu_scalar[grid](
                input,
                other,
                output,
                n_elements,
                alpha_value,
                ALPHA_IS_ONE=alpha_is_one,
                APPROXIMATE_TANH=approximate_tanh,
                BLOCK_SIZE=1024,
                num_warps=4,
            )
        return output

    return wrapper
