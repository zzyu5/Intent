import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


_BLOCK = 2048
_NUM_WARPS = 8


@triton.jit
def _tanh_contiguous_kernel(
    input,
    output,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(input + offsets)

    magnitude = tl.abs(values)
    decay = libdevice.fast_expf(-2.0 * magnitude)
    result = (1.0 - decay) / (1.0 + decay)
    result = tl.where(values < 0.0, -result, result)

    tl.store(output + offsets, result)


@triton.jit
def _tanh_contiguous_masked_kernel(
    input,
    output,
    n_elements,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input + offsets, mask=mask, other=0.0)

    magnitude = tl.abs(values)
    decay = libdevice.fast_expf(-2.0 * magnitude)
    result = (1.0 - decay) / (1.0 + decay)
    result = tl.where(values < 0.0, -result, result)

    tl.store(output + offsets, result, mask=mask)


@triton.jit
def _tanh_strided_kernel(
    input,
    output,
    n_elements,
    input_stride,
    output_stride,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input + offsets * input_stride, mask=mask, other=0.0)

    magnitude = tl.abs(values)
    decay = libdevice.fast_expf(-2.0 * magnitude)
    result = (1.0 - decay) / (1.0 + decay)
    result = tl.where(values < 0.0, -result, result)

    tl.store(output + offsets * output_stride, result, mask=mask)


def launch(input, output):
    n_elements = input.shape[0]
    grid = (triton.cdiv(n_elements, _BLOCK),)
    if input.stride(0) == 1 and output.stride(0) == 1:
        if n_elements % _BLOCK == 0:
            return _tanh_contiguous_kernel[(n_elements // _BLOCK,)](
                input,
                output,
                BLOCK=_BLOCK,
                num_warps=_NUM_WARPS,
            )
        return _tanh_contiguous_masked_kernel[grid](
            input,
            output,
            n_elements,
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )
    return _tanh_strided_kernel[grid](
        input,
        output,
        n_elements,
        input.stride(0),
        output.stride(0),
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )


def run(input):
    output = torch.empty((input.shape[0],), device=input.device, dtype=torch.float32)
    launch(input, output)
    return output
