import torch
import triton
import triton.language as tl


@triton.jit
def _sub_gelu_exact(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    alpha,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    input_value = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    other_value = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    value = input_value - alpha * other_value

    # Abramowitz-Stegun 7.1.26 gives a sufficiently accurate erf for the
    # exact GELU tolerance while avoiding backend-specific erf lowering.
    scaled = value * 0.7071067811865476
    absolute = tl.abs(scaled)
    t = 1.0 / (1.0 + 0.3275911 * absolute)
    polynomial = (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
                    - 0.284496736) * t + 0.254829592) * t)
    erf_value = tl.where(
        scaled < 0.0,
        -(1.0 - polynomial * tl.exp(-absolute * absolute)),
        1.0 - polynomial * tl.exp(-absolute * absolute),
    )
    result = 0.5 * value * (1.0 + erf_value)
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _sub_gelu_tanh(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    alpha,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    input_value = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    other_value = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    value = input_value - alpha * other_value
    cube = value * value * value
    inner = 0.7978845608028654 * (value + 0.044715 * cube)
    result = 0.5 * value * (1.0 + tl.tanh(inner))
    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def sub_gelu(input, other, alpha=1, approximate="none", out=None):
        if approximate not in ("none", "tanh"):
            raise ValueError("approximate must be 'none' or 'tanh'")

        output = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        kernel = _sub_gelu_exact if approximate == "none" else _sub_gelu_tanh
        kernel[grid](input, other, output, n_elements, alpha, BLOCK=1024, num_warps=4)
        return output

    return sub_gelu
