import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK": 256}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 512}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 2048}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 2048}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 4096}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 4096}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 8192}, num_warps=8, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def _relu_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    values = tl.maximum(values, 0.0, propagate_nan=tl.PropagateNan.ALL)
    tl.store(output_ptr + offsets, values, mask=mask)


def build(context):
    def wrapper(input, inplace=False):
        n_elements = input.numel()
        grid = lambda META: (triton.cdiv(n_elements, META["BLOCK"]),)
        if inplace:
            _relu_kernel[grid](input, input, n_elements)
            return input
        output = torch.empty_like(input)
        _relu_kernel[grid](input, output, n_elements)
        return output

    return wrapper
