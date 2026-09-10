import torch
import triton
import triton.language as tl


@triton.jit
def _sub_gelu_tensor(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    alpha: tl.constexpr,
    approximate: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    input_value = tl.load(input_ptr + offsets, mask=mask)
    other_value = tl.load(other_ptr + offsets, mask=mask)
    value = input_value - alpha * other_value

    if approximate:
        value = 0.5 * value * (
            1.0
            + tl.tanh(
                0.7978845608028654
                * (value + 0.044715 * value * value * value)
            )
        )
    else:
        value = 0.5 * value * (1.0 + tl.erf(value * 0.7071067811865475))

    tl.store(output_ptr + offsets, value, mask=mask)


@triton.jit
def _sub_gelu_scalar(
    input_ptr,
    output_ptr,
    n_elements,
    other: tl.constexpr,
    alpha: tl.constexpr,
    approximate: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    input_value = tl.load(input_ptr + offsets, mask=mask)
    value = input_value - alpha * other

    if approximate:
        value = 0.5 * value * (
            1.0
            + tl.tanh(
                0.7978845608028654
                * (value + 0.044715 * value * value * value)
            )
        )
    else:
        value = 0.5 * value * (1.0 + tl.erf(value * 0.7071067811865475))

    tl.store(output_ptr + offsets, value, mask=mask)


def build(context):
    del context

    def wrapper(input, other, alpha=1, approximate="none", out=None):
        if approximate == "none":
            approximate_flag = 0
        elif approximate == "tanh":
            approximate_flag = 1
        else:
            raise ValueError("approximate must be 'none' or 'tanh'")

        output = out if out is not None else torch.empty_like(input)

        n_elements = input.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        alpha_value = float(alpha)

        if isinstance(other, torch.Tensor):
            _sub_gelu_tensor[grid](
                input,
                other,
                output,
                n_elements,
                alpha=alpha_value,
                approximate=approximate_flag,
                BLOCK=1024,
                num_warps=4,
            )
        else:
            _sub_gelu_scalar[grid](
                input,
                output,
                n_elements,
                other=float(other),
                alpha=alpha_value,
                approximate=approximate_flag,
                BLOCK=1024,
                num_warps=4,
            )
        return output

    return wrapper
