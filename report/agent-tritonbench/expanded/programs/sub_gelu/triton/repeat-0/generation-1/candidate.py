import torch
import triton
import triton.language as tl


@triton.jit
def _sub_gelu_exact_tensor(
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
    result = 0.5 * value * (1.0 + tl.erf(value * 0.7071067811865476))
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _sub_gelu_tanh_tensor(
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


@triton.jit
def _sub_gelu_exact_scalar(
    input_ptr,
    other,
    output_ptr,
    n_elements,
    alpha,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    input_value = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    value = input_value - alpha * other
    result = 0.5 * value * (1.0 + tl.erf(value * 0.7071067811865476))
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _sub_gelu_tanh_scalar(
    input_ptr,
    other,
    output_ptr,
    n_elements,
    alpha,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    input_value = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    value = input_value - alpha * other
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

        if isinstance(other, torch.Tensor):
            kernel = _sub_gelu_exact_tensor if approximate == "none" else _sub_gelu_tanh_tensor
            kernel[grid](input, other, output, n_elements, alpha, BLOCK=1024, num_warps=4)
        else:
            kernel = _sub_gelu_exact_scalar if approximate == "none" else _sub_gelu_tanh_scalar
            kernel[grid](input, other, output, n_elements, alpha, BLOCK=1024, num_warps=4)
        return output

    return sub_gelu
