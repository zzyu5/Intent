import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 1024}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 2048}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK": 2048}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 2048}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 4096}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK": 4096}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 4096}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 512}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK": 512}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 512}, num_warps=8, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def abs_kernel(input, output, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    value = tl.load(input + offsets, mask=mask)
    tl.store(output + offsets, tl.abs(value), mask=mask)


def launch(input, output):
    n_elements = input.numel()
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK"]),)
    return abs_kernel[grid](input, output, n_elements)


def run(input):
    output = torch.empty((input.shape[0],), device=input.device, dtype=input.dtype)
    launch(input, output)
    return output
