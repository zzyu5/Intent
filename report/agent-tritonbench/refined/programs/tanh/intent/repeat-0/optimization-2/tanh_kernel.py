import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK": 256}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 512}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK": 512}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 2048}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 2048}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def _tanh_kernel(
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

    result = libdevice.tanh(values)

    tl.store(output + offsets * output_stride, result, mask=mask)


def launch(input, output):
    n_elements = input.shape[0]
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)
    return _tanh_kernel[grid](
        input,
        output,
        n_elements,
        input.stride(0),
        output.stride(0),
    )


def run(input):
    output = torch.empty((input.shape[0],), device=input.device, dtype=torch.float32)
    launch(input, output)
    return output
