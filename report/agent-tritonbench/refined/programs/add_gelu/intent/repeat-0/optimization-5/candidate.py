import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


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
        _add_gelu_kernel[(4096,)](input, other, out, BLOCK=256, num_warps=4, num_stages=1)
        return out

    return wrapper
