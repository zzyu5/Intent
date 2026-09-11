import torch
import triton
import triton.language as tl


_FIXED_NUM_ELEMENTS = 1048576


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 128}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK": 128}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 256}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK": 256}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 512}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 2048}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 2048}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 4096}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 8192}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 8192}, num_warps=8, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def _exp_contiguous(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(input_ptr + offsets)
    result = tl.exp(values)
    tl.store(output_ptr + offsets, result)


@triton.jit
def _exp_strided(input_ptr, output_ptr, n_elements, input_stride, output_stride, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets * input_stride, mask=mask)
    result = tl.exp(values)
    tl.store(output_ptr + offsets * output_stride, result, mask=mask)


def build(context):
    def wrapper(input, *, out=None):
        if out is None:
            out = torch.empty_like(input)
        n_elements = input.numel()
        if (n_elements == _FIXED_NUM_ELEMENTS and input.stride(0) == 1 and out.stride(0) == 1):
            grid = lambda meta: (_FIXED_NUM_ELEMENTS // meta["BLOCK"],)
            _exp_contiguous[grid](input, out, n_elements)
        else:
            grid = (triton.cdiv(n_elements, 256),)
            _exp_strided[grid](input, out, n_elements, input.stride(0), out.stride(0), BLOCK=256)
        return out

    return wrapper
