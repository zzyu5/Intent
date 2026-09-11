import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=2),
        triton.Config({"BLOCK": 256}, num_warps=4),
        triton.Config({"BLOCK": 256}, num_warps=8),
        triton.Config({"BLOCK": 512}, num_warps=2),
        triton.Config({"BLOCK": 512}, num_warps=4),
        triton.Config({"BLOCK": 512}, num_warps=8),
        triton.Config({"BLOCK": 512}, num_warps=16),
        triton.Config({"BLOCK": 1024}, num_warps=2),
        triton.Config({"BLOCK": 1024}, num_warps=4),
        triton.Config({"BLOCK": 1024}, num_warps=8),
        triton.Config({"BLOCK": 1024}, num_warps=16),
        triton.Config({"BLOCK": 2048}, num_warps=4),
        triton.Config({"BLOCK": 2048}, num_warps=8),
        triton.Config({"BLOCK": 2048}, num_warps=16),
        triton.Config({"BLOCK": 4096}, num_warps=8),
        triton.Config({"BLOCK": 4096}, num_warps=16),
    ],
    key=["n_elements"],
)
@triton.jit
def log1p_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(input_ptr + offsets)
    tl.store(output_ptr + offsets, libdevice.log1p(values))


def build(context):
    def wrapper(input, *, out=None):
        if out is None:
            out = torch.empty_like(input)
        n_elements = input.numel()
        if n_elements:
            log1p_kernel[
                lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)
            ](input, out, n_elements)
        return out

    return wrapper
