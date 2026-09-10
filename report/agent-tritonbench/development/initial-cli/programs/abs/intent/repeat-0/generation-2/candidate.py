import torch
import intent
from intent import language as tl


@intent.kernel
def abs_kernel(x, n_elements, out, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask, other=0)
    tl.store(out + offsets, tl.abs(values), mask=mask)


def build(context):
    block_size = 256
    kernel = context.compile(
        "abs_elementwise_kernel",
        abs_kernel,
        constexprs={"BLOCK_SIZE": block_size},
    )

    def wrapper(x):
        out = torch.empty_like(x)
        kernel(x, x.numel(), out)
        return out

    return wrapper
