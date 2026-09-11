import torch
import triton
import triton.language as tl


@triton.jit
def asin_contiguous_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # A minimax-style approximation with a square-root endpoint transform.
    abs_values = tl.abs(values)
    polynomial = (((-0.0187293 * abs_values + 0.0742610) * abs_values - 0.2121144) * abs_values
                  + 1.5707288)
    result = 1.5707963267948966 - tl.sqrt(1.0 - abs_values) * polynomial
    result = tl.where(values < 0.0, -result, result)
    tl.store(output_ptr + offsets, result.to(tl.float16), mask=mask)


@triton.jit
def asin_strided_kernel(input_ptr, output_ptr, n_elements, input_stride, output_stride, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets * input_stride, mask=mask, other=0.0).to(tl.float32)
    abs_values = tl.abs(values)
    polynomial = (((-0.0187293 * abs_values + 0.0742610) * abs_values - 0.2121144) * abs_values
                  + 1.5707288)
    result = 1.5707963267948966 - tl.sqrt(1.0 - abs_values) * polynomial
    result = tl.where(values < 0.0, -result, result)
    tl.store(output_ptr + offsets * output_stride, result.to(tl.float16), mask=mask)


def build(context):
    def wrapper(input, *, out=None):
        output = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        if input.stride(0) == 1 and output.stride(0) == 1:
            asin_contiguous_kernel[(triton.cdiv(n_elements, 1024),)](
                input, output, n_elements, BLOCK=1024, num_warps=8
            )
        else:
            asin_strided_kernel[(triton.cdiv(n_elements, 1024),)](
                input,
                output,
                n_elements,
                input.stride(0),
                output.stride(0),
                BLOCK=1024,
                num_warps=8,
            )
        return output

    return wrapper
