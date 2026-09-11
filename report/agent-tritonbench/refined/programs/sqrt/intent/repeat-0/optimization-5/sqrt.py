import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _sqrt_contiguous_fixed(input_ptr, output_ptr):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    values = tl.load(input_ptr + offsets)
    tl.store(output_ptr + offsets, tl.sqrt(values))


@triton.jit
def _sqrt_contiguous(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    result = libdevice.sqrt(values)
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _sqrt_strided(
    input_ptr,
    output_ptr,
    n_elements,
    input_stride,
    output_stride,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets * input_stride, mask=mask, other=0.0)
    result = libdevice.sqrt(values)
    tl.store(output_ptr + offsets * output_stride, result, mask=mask)


def launch(input, output):
    n_elements = input.numel()
    if input.stride(0) == 1 and output.stride(0) == 1:
        if n_elements == 1048576:
            return _sqrt_contiguous_fixed[(4096,)](
                input,
                output,
                num_warps=1,
                num_stages=1,
            )
        grid = (triton.cdiv(n_elements, 256),)
        return _sqrt_contiguous[grid](
            input,
            output,
            n_elements,
            BLOCK_SIZE=256,
            num_warps=1,
            num_stages=1,
        )

    grid = (triton.cdiv(n_elements, 256),)
    return _sqrt_strided[grid](
        input,
        output,
        n_elements,
        input.stride(0),
        output.stride(0),
        BLOCK_SIZE=256,
    )


def run(input):
    output = torch.empty((input.shape[0],), device=input.device, dtype=torch.float32)
    launch(input, output)
    return output
