import torch
import triton
import triton.language as tl


@triton.jit
def _rsqrt_fixed(input, output):
    offsets = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    values = tl.load(input + offsets)
    tl.store(output + offsets, tl.rsqrt(values))


@triton.jit
def _rsqrt_strided(
    input,
    output,
    n_elements,
    input_stride,
    output_stride,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input + offsets * input_stride, mask=mask)
    tl.store(output + offsets * output_stride, tl.rsqrt(values), mask=mask)


def launch(input, output):
    n_elements = input.numel()
    if (
        n_elements == 1048576
        and input.stride(0) == 1
        and output.stride(0) == 1
    ):
        return _rsqrt_fixed[(1024,)](input, output, num_warps=4, num_stages=1)

    grid = (triton.cdiv(n_elements, 256),)
    return _rsqrt_strided[
        grid
    ](input, output, n_elements, input.stride(0), output.stride(0), BLOCK=256)


def run(input):
    output = torch.empty((input.shape[0],), device=input.device, dtype=torch.float32)
    launch(input, output)
    return output
