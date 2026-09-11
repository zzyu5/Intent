import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 128}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 256}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 512}, num_warps=1, num_stages=1),
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
    ],
    key=[],
)
@triton.jit
def _add_gelu_kernel(input, other, output, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(input + offsets).to(tl.float32)
    y = tl.load(other + offsets).to(tl.float32)
    x = x + y

    x2 = x * x
    inner = 0.7978845834732056 * (x + 0.044714998453855515 * (x2 * x))
    result = 0.5 * x * (1.0 + libdevice.tanh(inner))
    tl.store(output + offsets, result.to(tl.float16))


def build(context):
    def wrapper(input, other, alpha=1, approximate='none', out=None):
        if out is None:
            out = torch.empty_like(input)
        grid = lambda META: (triton.cdiv(input.numel(), META["BLOCK"]),)
        _add_gelu_kernel[grid](input, other, out)
        return out

    return wrapper
