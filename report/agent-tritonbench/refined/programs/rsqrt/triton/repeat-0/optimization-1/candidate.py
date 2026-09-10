import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=2),
        triton.Config({"BLOCK": 256}, num_warps=4),
        triton.Config({"BLOCK": 512}, num_warps=2),
        triton.Config({"BLOCK": 512}, num_warps=4),
        triton.Config({"BLOCK": 512}, num_warps=8),
        triton.Config({"BLOCK": 1024}, num_warps=2),
        triton.Config({"BLOCK": 1024}, num_warps=4),
        triton.Config({"BLOCK": 1024}, num_warps=8),
        triton.Config({"BLOCK": 2048}, num_warps=4),
        triton.Config({"BLOCK": 2048}, num_warps=8),
        triton.Config({"BLOCK": 2048}, num_warps=16),
        triton.Config({"BLOCK": 4096}, num_warps=8),
        triton.Config({"BLOCK": 4096}, num_warps=16),
        triton.Config({"BLOCK": 8192}, num_warps=8),
        triton.Config({"BLOCK": 8192}, num_warps=16),
        triton.Config({"BLOCK": 16384}, num_warps=16),
    ],
    key=["n_elements"],
)
@triton.jit
def rsqrt_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, tl.rsqrt(values), mask=mask)


def build(context):
    def wrapper(input, *, out=None):
        output = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        rsqrt_kernel[lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)](
            input, output, n_elements
        )
        return output

    return wrapper
