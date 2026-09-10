import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def _sqrt_contiguous(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    result = libdevice.sqrt(values)
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def _sqrt_contiguous_full(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(input_ptr + offsets)
    result = libdevice.sqrt(values)
    tl.store(output_ptr + offsets, result)


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
        grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
        if n_elements == 1048576:
            return _sqrt_contiguous_full[grid](input, output, n_elements)
        return _sqrt_contiguous[grid](input, output, n_elements)

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
