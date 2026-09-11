import torch
import triton
import triton.language as tl


@triton.jit
def _mul_relu_tensor(input_ptr, other_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    input_value = tl.load(input_ptr + offsets, mask=mask)
    other_value = tl.load(other_ptr + offsets, mask=mask)
    result = tl.maximum(input_value * other_value, 0.0)
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _mul_relu_tensor_full(input_ptr, other_ptr, output_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    input_value = tl.load(input_ptr + offsets)
    other_value = tl.load(other_ptr + offsets)
    result = tl.maximum(input_value * other_value, 0.0)
    tl.store(output_ptr + offsets, result)


@triton.jit
def _mul_relu_scalar(input_ptr, other_value, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    input_value = tl.load(input_ptr + offsets, mask=mask)
    result = tl.maximum(input_value * other_value, 0.0)
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _mul_relu_scalar_tensor(input_ptr, other_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    input_value = tl.load(input_ptr + offsets, mask=mask)
    other_value = tl.load(other_ptr)
    result = tl.maximum(input_value * other_value, 0.0)
    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def wrapper(input, other, inplace=False, out=None):
        if out is not None:
            output = out
        elif inplace:
            output = input
        else:
            output = torch.empty_like(input)

        count = input.numel()
        if torch.is_tensor(other):
            if other.numel() == 1:
                grid = (triton.cdiv(count, 1024),)
                _mul_relu_scalar_tensor[grid](input, other, output, count, BLOCK=1024, num_warps=4)
            elif count == 1048576:
                _mul_relu_tensor_full[(512,)](input, other, output, BLOCK=2048, num_warps=4, num_stages=1)
            elif count % 1024 == 0:
                _mul_relu_tensor_full[(count // 1024,)](input, other, output, BLOCK=1024, num_warps=4)
            else:
                grid = (triton.cdiv(count, 1024),)
                _mul_relu_tensor[grid](input, other, output, count, BLOCK=1024, num_warps=4)
        else:
            grid = (triton.cdiv(count, 1024),)
            _mul_relu_scalar[grid](input, other, output, count, BLOCK=1024, num_warps=4)
        return output

    return wrapper
